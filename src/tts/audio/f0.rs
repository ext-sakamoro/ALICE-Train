//! YIN algorithm による F0 (fundamental frequency) 推定。
//!
//! de Cheveigné & Kawahara (2002) の YIN 論文をベースにした古典的 F0 抽出。
//! WORLD (DIO/StoneMask) や CREPE と比較すると精度は劣るが、Rust ゼロ依存で実装可能。
//!
//! # 参考
//!
//! - de Cheveigné, A., & Kawahara, H. (2002). "YIN, a fundamental frequency estimator for speech and music"
//! - librosa `librosa.yin()` と類似アルゴリズム (パラメータ範囲は概ね一致)
//!
//! # V/UV 判定
//!
//! YIN cumulative mean normalized difference 関数の最小値が threshold (通常 0.15) を
//! 超えた場合、そのフレームは unvoiced として `F0 = 0`, `voiced = false` を返す。

/// YIN algorithm による frame-wise F0 抽出。
///
/// # 引数
///
/// - `wav`: 入力波形 (mono f32)
/// - `sample_rate`: サンプリング周波数 (Hz)
/// - `frame_length`: 各 frame の解析窓長 (通常 `2048`、hop の 4 倍以上推奨)
/// - `hop_length`: hop サイズ (通常 `256` @ 24 kHz)
/// - `f_min`: 探索最低周波数 (Hz、通常 50.0)
/// - `f_max`: 探索最高周波数 (Hz、通常 550.0 for speech)
/// - `threshold`: YIN CMND 閾値 (通常 0.15、低すぎると unvoiced 判定増える)
///
/// # 戻り値
///
/// `(Vec<f32>, Vec<bool>)`:
/// - `Vec<f32>`: F0 in Hz (unvoiced frame は 0.0)
/// - `Vec<bool>`: voiced (true) / unvoiced (false) flag
///
/// 両 Vec の長さは frame 数 = `1 + wav.len() / hop_length` (STFT と同じ frame 定義)。
///
/// # Panics
///
/// - `wav.is_empty()`
/// - `hop_length == 0`
/// - `frame_length < hop_length * 2`
/// - `f_min >= f_max`
/// - `f_min < sample_rate / frame_length` (探索範囲が窓長を超える)
#[must_use]
pub fn yin_f0(
    wav: &[f32],
    sample_rate: u32,
    frame_length: usize,
    hop_length: usize,
    f_min: f32,
    f_max: f32,
    threshold: f32,
) -> (Vec<f32>, Vec<bool>) {
    assert!(!wav.is_empty(), "wav must not be empty");
    assert!(hop_length > 0, "hop_length must be > 0");
    assert!(
        frame_length >= hop_length * 2,
        "frame_length must be >= hop_length * 2 for reasonable YIN accuracy"
    );
    assert!(f_min < f_max, "f_min must be < f_max");
    assert!(
        f_min * frame_length as f32 >= sample_rate as f32,
        "f_min {f_min} Hz too low for frame_length {frame_length} @ sr {sample_rate}"
    );

    let sr = sample_rate as f32;
    let tau_max = (sr / f_min).ceil() as usize;
    let tau_min = (sr / f_max).floor() as usize;
    let tau_min = tau_min.max(1);

    // center padding (STFT と frame 数を揃えるため)
    let pad = frame_length / 2;
    let padded = center_pad(wav, pad);

    let n_frames = 1 + wav.len() / hop_length;
    let mut f0 = Vec::with_capacity(n_frames);
    let mut voiced = Vec::with_capacity(n_frames);

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_length;
        if start + frame_length > padded.len() {
            f0.push(0.0);
            voiced.push(false);
            continue;
        }

        let frame = &padded[start..start + frame_length];
        let (est_f0, is_voiced) = estimate_f0_frame(frame, sr, tau_min, tau_max, threshold);
        f0.push(est_f0);
        voiced.push(is_voiced);
    }

    (f0, voiced)
}

/// 単一 frame の F0 を YIN で推定する。
fn estimate_f0_frame(
    frame: &[f32],
    sample_rate: f32,
    tau_min: usize,
    tau_max: usize,
    threshold: f32,
) -> (f32, bool) {
    let w = frame.len();
    let tau_max = tau_max.min(w / 2);
    if tau_max <= tau_min {
        return (0.0, false);
    }

    // Step 1: difference function d(τ) = Σ (x[j] - x[j+τ])²
    let mut d = vec![0.0_f32; tau_max + 1];
    for tau in 1..=tau_max {
        let n = w - tau;
        let mut acc = 0.0_f32;
        for j in 0..n {
            let diff = frame[j] - frame[j + tau];
            acc += diff * diff;
        }
        d[tau] = acc;
    }

    // Step 2: cumulative mean normalized difference d'(τ)
    let mut d_prime = vec![1.0_f32; tau_max + 1];
    let mut running_sum = 0.0_f32;
    for tau in 1..=tau_max {
        running_sum += d[tau];
        if running_sum > 0.0 {
            d_prime[tau] = d[tau] * (tau as f32) / running_sum;
        } else {
            d_prime[tau] = 1.0;
        }
    }

    // Step 3: absolute threshold — search τ in [tau_min, tau_max]
    let mut tau_est: Option<usize> = None;
    for tau in tau_min..=tau_max {
        if d_prime[tau] < threshold {
            // 局所最小になるまで tau を進める
            let mut t = tau;
            while t < tau_max && d_prime[t + 1] < d_prime[t] {
                t += 1;
            }
            tau_est = Some(t);
            break;
        }
    }

    let Some(tau) = tau_est else {
        // 閾値以下なし → unvoiced
        return (0.0, false);
    };

    // Step 4: parabolic interpolation で sub-sample 精度
    let refined_tau = parabolic_interpolate(&d_prime, tau);
    if refined_tau <= 0.0 {
        return (0.0, false);
    }
    let f0 = sample_rate / refined_tau;

    (f0, true)
}

/// 3 点 parabolic interpolation で sub-sample τ を求める。
fn parabolic_interpolate(d_prime: &[f32], tau: usize) -> f32 {
    if tau == 0 || tau >= d_prime.len() - 1 {
        return tau as f32;
    }
    let a = d_prime[tau - 1];
    let b = d_prime[tau];
    let c = d_prime[tau + 1];
    let denom = a - 2.0 * b + c;
    if denom.abs() < 1e-9 {
        return tau as f32;
    }
    let offset = 0.5 * (a - c) / denom;
    (tau as f32) + offset
}

/// center padding: 波形の両端に `pad` サンプル 0 埋め。
fn center_pad(wav: &[f32], pad: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(wav.len() + 2 * pad);
    out.extend(std::iter::repeat_n(0.0_f32, pad));
    out.extend_from_slice(wav);
    out.extend(std::iter::repeat_n(0.0_f32, pad));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 440 Hz sine wave の F0 が中央 frame で 440 Hz に十分近いか
    #[test]
    fn yin_detects_440hz_sine() {
        let sr = 24_000_u32;
        let freq = 440.0_f32;
        let n_samples = 24_000;
        let wav: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * freq * (i as f32) / sr as f32).sin())
            .collect();

        let (f0, voiced) = yin_f0(&wav, sr, 2048, 256, 50.0, 800.0, 0.15);

        // 中央 frame は必ず voiced
        let mid = f0.len() / 2;
        assert!(voiced[mid], "middle frame should be voiced");

        // ±3% 誤差以内
        let err = (f0[mid] - freq).abs() / freq;
        assert!(
            err < 0.03,
            "f0={} expected={} err={:.4}",
            f0[mid],
            freq,
            err
        );
    }

    #[test]
    fn yin_detects_200hz_sine() {
        let sr = 24_000_u32;
        let freq = 200.0_f32;
        let n_samples = 24_000;
        let wav: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * freq * (i as f32) / sr as f32).sin())
            .collect();

        let (f0, voiced) = yin_f0(&wav, sr, 2048, 256, 50.0, 800.0, 0.15);
        let mid = f0.len() / 2;
        assert!(voiced[mid]);
        let err = (f0[mid] - freq).abs() / freq;
        assert!(
            err < 0.03,
            "f0={} expected={} err={:.4}",
            f0[mid],
            freq,
            err
        );
    }

    #[test]
    fn yin_silence_is_unvoiced() {
        let wav = vec![0.0_f32; 24_000];
        let (f0, voiced) = yin_f0(&wav, 24_000, 2048, 256, 50.0, 800.0, 0.15);

        // 中央付近は全 unvoiced (F0 = 0)
        // Note: 冒頭/末尾 frame は padding の影響で voiced 判定になる可能性ゼロではない
        let mid = f0.len() / 2;
        assert!(
            !voiced[mid],
            "silence middle frame should be unvoiced, got voiced={} f0={}",
            voiced[mid], f0[mid]
        );
        assert!(f0[mid].abs() < f32::EPSILON);
    }

    #[test]
    fn yin_output_lengths_match() {
        let wav = vec![0.0_f32; 24_000];
        let (f0, voiced) = yin_f0(&wav, 24_000, 2048, 256, 50.0, 800.0, 0.15);
        assert_eq!(f0.len(), voiced.len());
        assert_eq!(f0.len(), 1 + 24_000 / 256);
    }

    #[test]
    fn parabolic_interpolate_at_symmetric_minimum() {
        // d_prime[2] が最小、対称なら refined = 2.0 (offset = 0)
        let d = vec![0.5, 0.3, 0.1, 0.3, 0.5];
        let refined = parabolic_interpolate(&d, 2);
        assert!((refined - 2.0).abs() < 1e-6);
    }

    #[test]
    fn parabolic_interpolate_shifts_toward_lower_neighbor() {
        // d_prime[2] が最小、右が左より小さいなら refined > 2.0
        let d = vec![0.5, 0.3, 0.1, 0.2, 0.5];
        let refined = parabolic_interpolate(&d, 2);
        assert!(refined > 2.0, "refined={refined}");
        assert!(refined < 3.0);
    }

    #[test]
    #[should_panic(expected = "wav must not be empty")]
    fn yin_panics_on_empty_wav() {
        let _ = yin_f0(&[], 24_000, 2048, 256, 50.0, 800.0, 0.15);
    }

    #[test]
    #[should_panic(expected = "frame_length must be >= hop_length * 2")]
    fn yin_panics_on_frame_shorter_than_hop_x2() {
        let wav = vec![0.0_f32; 1000];
        let _ = yin_f0(&wav, 24_000, 100, 256, 50.0, 800.0, 0.15);
    }
}
