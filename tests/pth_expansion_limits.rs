//! An entry of a `.pth` archive is decompressed only up to a limit, whatever its header declares.
//!
//! A stored entry (what `torch.save` writes) cannot expand past the archive, so its limit is its stored size; a
//! compressed entry is where a few kilobytes can expand to gigabytes, so it is read up to 256 MiB. A "bomb" below is
//! a deflated entry of 300 MiB of zeros (a few hundred kilobytes in the archive), written in chunks so the test never
//! holds the expanded bytes.

use alice_train::pth_loader::PthLoader;
use std::io::{Cursor, Write};
use std::path::PathBuf;

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/pth_zip")
            .join(name),
    )
    .unwrap()
}

/// The archive with entry `target` replaced by a deflated stream of `n` zero bytes; every other entry copied as is
fn with_bomb(src: &[u8], target: &str, n: usize) -> Vec<u8> {
    let mut archive = zip::ZipArchive::new(Cursor::new(src)).unwrap();
    let mut out = zip::ZipWriter::new(Cursor::new(Vec::new()));
    let mut found = false;
    for i in 0..archive.len() {
        let entry = archive.by_index_raw(i).unwrap();
        if entry.name().unwrap() == target {
            found = true;
            drop(entry);
            let opts = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            out.start_file(target, opts).unwrap();
            let chunk = vec![0u8; 1 << 20];
            let mut left = n;
            while left > 0 {
                let k = left.min(chunk.len());
                out.write_all(&chunk[..k]).unwrap();
                left -= k;
            }
        } else {
            out.raw_copy_file(entry).unwrap();
        }
    }
    assert!(found, "{target} is in the archive");
    out.finish().unwrap().into_inner()
}

const BOMB: usize = 300 << 20;

#[test]
fn a_compressed_storage_that_expands_past_the_limit_is_not_read() {
    let src = fixture("stored.pth");
    let loader = PthLoader::from_bytes(src.clone()).unwrap();
    let key = loader.tensor_meta("w").unwrap().storage_key.clone();
    let bomb = with_bomb(&src, &format!("archive/data/{key}"), BOMB);
    assert!(
        bomb.len() < BOMB / 100,
        "the bomb is small: {} bytes",
        bomb.len()
    );
    let loader = PthLoader::from_bytes(bomb).unwrap();
    assert!(
        loader.get_tensor_raw("w").is_none(),
        "the bomb's storage is refused"
    );
    assert!(loader.get_tensor_f32("w").is_none());
    // a tensor in another storage still reads
    assert!(loader.get_tensor_raw("b").is_some());
}

#[test]
fn a_compressed_data_pkl_that_expands_past_the_limit_is_refused() {
    let bomb = with_bomb(&fixture("stored.pth"), "archive/data.pkl", BOMB);
    let err = PthLoader::from_bytes(bomb).err().expect("refused");
    assert!(err.to_string().contains("limit"), "{err}");
}

/// `bytes` with every record of `name` (local file header and central directory header) declaring `size` as its
/// uncompressed size: the archive lies about how far the entry expands
fn declaring(mut bytes: Vec<u8>, name: &str, size: u32) -> Vec<u8> {
    let mut patched = 0;
    for (sig, size_at, name_len_at, name_at) in [
        (b"PK\x03\x04", 22usize, 26usize, 30usize),
        (b"PK\x01\x02", 24, 28, 46),
    ] {
        let mut i = 0;
        while let Some(p) = bytes[i..].windows(4).position(|w| w == sig) {
            let at = i + p;
            let n =
                u16::from_le_bytes([bytes[at + name_len_at], bytes[at + name_len_at + 1]]) as usize;
            if bytes.get(at + name_at..at + name_at + n) == Some(name.as_bytes()) {
                bytes[at + size_at..at + size_at + 4].copy_from_slice(&size.to_le_bytes());
                patched += 1;
            }
            i = at + 4;
        }
    }
    assert_eq!(patched, 2, "both records of {name} patched");
    bytes
}

#[test]
fn an_entry_that_declares_a_small_size_and_expands_past_the_limit_is_not_read() {
    // the declared size passes the check, so the read is what must stop: the loader caps it at the limit, and zip 9
    // also stops at the declared size, so this pins that the two together refuse a lying entry (a zip upgrade that
    // read past the declared size would be caught by the loader's own cap)
    let src = fixture("stored.pth");
    let key = PthLoader::from_bytes(src.clone())
        .unwrap()
        .tensor_meta("w")
        .unwrap()
        .storage_key
        .clone();
    let name = format!("archive/data/{key}");
    let bomb = declaring(with_bomb(&src, &name, BOMB), &name, 1000);
    let loader = PthLoader::from_bytes(bomb).unwrap();
    assert!(
        loader.get_tensor_raw("w").is_none(),
        "the lying entry is refused"
    );
    assert!(loader.get_tensor_raw("b").is_some());
}

/// The archive with `n` more deflated entries of 16 bytes each (`archive/extra/<k>`), every original entry copied
fn with_extra(src: &[u8], n: usize) -> Vec<u8> {
    let mut archive = zip::ZipArchive::new(Cursor::new(src)).unwrap();
    let mut out = zip::ZipWriter::new(Cursor::new(Vec::new()));
    for i in 0..archive.len() {
        out.raw_copy_file(archive.by_index_raw(i).unwrap()).unwrap();
    }
    for k in 0..n {
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        out.start_file(format!("archive/extra/{k}"), opts).unwrap();
        out.write_all(&[k as u8; 16]).unwrap();
    }
    out.finish().unwrap().into_inner()
}

#[test]
fn compressed_entries_that_declare_more_than_the_archive_total_are_refused() {
    // each entry declares just under the per-entry limit, so only the archive-wide total can refuse them: five of
    // them declare 1275 MiB, past the 1 GiB the loader allows the compressed entries of one archive together
    let per_entry = (256u32 << 20) - (1 << 20);
    let mut bomb = with_extra(&fixture("stored.pth"), 5);
    for k in 0..5 {
        bomb = declaring(bomb, &format!("archive/extra/{k}"), per_entry);
    }
    let err = PthLoader::from_bytes(bomb).err().expect("refused");
    assert!(err.to_string().contains("total"), "{err}");
    // four of them (1020 MiB) are within the total
    let mut fine = with_extra(&fixture("stored.pth"), 4);
    for k in 0..4 {
        fine = declaring(fine, &format!("archive/extra/{k}"), per_entry);
    }
    assert!(PthLoader::from_bytes(fine).is_ok());
}
