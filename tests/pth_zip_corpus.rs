//! `PthLoader` reads every archive of tests/fixtures/pth_zip the way it was
//! written.
//!
//! oracle: tests/fixtures/pth_zip/expected.tsv, computed by
//! scripts/gen_pth_zip_corpus.py from the bytes it wrote (not by this crate):
//! per archive the set of tensor names, and per tensor the dtype, the shape and
//! the SHA-256 of its bytes. The archives cover stored and deflated entries,
//! entry data padded to 64 bytes by an extra field, a storage of several
//! deflate blocks, two tensors sharing one storage at an offset, storages
//! before data.pkl with directory entries and an archive comment, and ZIP64
//! records, so a change of the ZIP reader that alters any of them is red.

use alice_train::pth_loader::PthLoader;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::PathBuf;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pth_zip")
}

/// archive -> tensor -> (dtype, shape, sha256)
fn expected() -> BTreeMap<String, BTreeMap<String, (String, String, String)>> {
    let text = std::fs::read_to_string(dir().join("expected.tsv")).expect("expected.tsv");
    let mut out: BTreeMap<String, BTreeMap<String, (String, String, String)>> = BTreeMap::new();
    for line in text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(f.len(), 5, "malformed line: {line}");
        out.entry(f[0].to_string()).or_default().insert(
            f[1].to_string(),
            (f[2].to_string(), f[3].to_string(), f[4].to_string()),
        );
    }
    out
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut s = String::new();
    for b in Sha256::digest(bytes) {
        write!(s, "{b:02x}").unwrap();
    }
    s
}

#[test]
fn every_archive_reads_as_written() {
    let want = expected();
    assert!(want.len() >= 7, "the corpus lists {} archives", want.len());
    let on_disk: BTreeSet<String> = std::fs::read_dir(dir())
        .unwrap()
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|n| n.ends_with(".pth"))
        .collect();
    assert_eq!(
        on_disk,
        want.keys().cloned().collect(),
        "archives on disk vs expected.tsv"
    );

    let mut compared = 0;
    for (archive, tensors) in &want {
        let loader =
            PthLoader::open(dir().join(archive)).unwrap_or_else(|e| panic!("{archive}: {e}"));
        let names: BTreeSet<String> = loader.tensor_names().into_iter().collect();
        assert_eq!(
            names,
            tensors.keys().cloned().collect(),
            "{archive}: tensor names"
        );
        for (name, (dtype, shape, digest)) in tensors {
            let (raw, got_dtype, got_shape) = loader
                .get_tensor_raw(name)
                .unwrap_or_else(|| panic!("{archive}/{name}: no bytes"));
            assert_eq!(format!("{got_dtype:?}"), *dtype, "{archive}/{name}: dtype");
            let got_shape = got_shape
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("x");
            assert_eq!(got_shape, *shape, "{archive}/{name}: shape");
            assert_eq!(sha256_hex(&raw), *digest, "{archive}/{name}: bytes");
            compared += 1;
        }
    }
    assert!(compared >= 15, "compared {compared} tensors");
}

#[test]
fn a_truncated_archive_is_an_error_not_a_panic() {
    let data = std::fs::read(dir().join("deflated.pth")).unwrap();
    for cut in [0, 10, data.len() / 2, data.len() - 1] {
        assert!(
            PthLoader::from_bytes(data[..cut].to_vec()).is_err(),
            "deflated.pth cut at {cut} of {} bytes was accepted",
            data.len()
        );
    }
}
