# ALICE-radical TTS Architecture

**Version**: draft-0.1 (2026-07-30)
**Anchor**: ALICE 原則「データを送るな、法則を送れ」("Don't ship data. Ship laws.")
**Reference**: [ALICE-Font v0.4.x](https://github.com/ext-sakamoro/ALICE-Font) — TTF outline coord を廃止し 40-byte `MetaFontParams` + parametric radical library に radical rewrite した先例

## 0. Why this document

現行 ALICE-TTS 実装 (FastSpeech2 acoustic + Griffin-Lim / Vocos vocoder) は **"pragmatic
learned law"** 側に位置し、ALICE 原則の spectrum のうち radical 端に到達していない。
ALICE-Font が v0.3.x (BIZ UDPGothic outline coord ~1MB を Rust const 埋込) →
v0.4.0 (40-byte MetaFontParams + parametric radical library 200KB total) の
subtractive rewrite で示した radical spirit を、TTS ドメインに適用する場合の
architecture を定義する。

**核心 insight** (2026-07-30 user 指摘):
> 声質・感情は後付け post-process 可能なので、base model は極小化できる。
> 「voice-baked」「emotion-baked」な巨大 model は ALICE 原則違反。

## 1. Layered decomposition

```
┌─────────────────────────────────────────────────┐
│ Layer 4: Vocoder (Vocos or 同等, 固定 backbone) │
│   mel → waveform、~10MB shared                  │
├─────────────────────────────────────────────────┤
│ Layer 3: Emotion post-process (40 bytes)        │
│   EmotionParams: pitch dynamics/tempo/energy    │
│   → mel を math で modulate、ゼロ学習           │
├─────────────────────────────────────────────────┤
│ Layer 2: Voice post-process (40 bytes)          │
│   VoiceParams: formant shift/pitch/timbre/breath │
│   → mel を math で warp、ゼロ学習               │
├─────────────────────────────────────────────────┤
│ Layer 1: Neutral acoustic base (~1MB)           │
│   Phoneme + prosody → 話者中立 mel              │
│   → 極小 neural (phonetic model)                │
└─────────────────────────────────────────────────┘
```

### Layer 責務分離

| Layer | 責務 | 実装形態 | 学習性 |
|-------|------|---------|-------|
| 1 (base) | Phoneme sequence → 話者中立 mel、prosody target 反映 | Neural (small FastSpeech2、~1.5M param) | learned once |
| 2 (voice) | Formant/pitch/timbre で話者性 injection | Pure math on mel | zero learning |
| 3 (emotion) | Pitch dynamics/tempo/energy で感情 injection | Pure math on mel | zero learning |
| 4 (vocoder) | mel → waveform 変換 | Neural (Vocos、fixed pretrained) | fixed |

### Payload budget

| 構成 | Payload | Coverage |
|------|---------|---------|
| Neutral base (FastSpeech2 hidden=128 layers=2) | ~1MB | 無限 text |
| VoiceParams library (100 preset × 40B) | ~4KB | 100 voice |
| EmotionParams library (20 preset × 40B) | ~800B | 20 emotion |
| Vocos vocoder (fixed) | ~10MB (shared) | 全音声 |
| **合計 (Vocos 除く)** | **~1.005MB** | **無限 (voice × emotion × text)** |
| **合計 (Vocos 込)** | **~11MB** | 同上 |

### 対比 (traditional multi-speaker × multi-emotion TTS)

- Per-speaker × per-emotion 全部 bake: 9MB × 100 speaker × 20 emotion = **~18GB**
- ALICE decomposed: **~11MB** で同 coverage (~1,600× 圧縮)

## 2. VoiceParams schema (40 bytes)

```rust
/// 話者性を parametric 表現する 40-byte struct
/// (ALICE-Font MetaFontParams と同精神)
#[repr(C)]
pub struct VoiceParams {
    /// F0 中央値 (Hz)、male ~120、female ~220、child ~280
    pub f0_mean_hz: f32,
    /// F0 range multiplier、monotone → 1.0、expressive → 1.5+
    pub f0_range_scale: f32,
    /// Formant frequency scale、male → 0.85、female → 1.0、child → 1.15
    pub formant_scale: f32,
    /// F1 offset (Hz)、母音空間の縦軸微調整
    pub formant_f1_offset_hz: f32,
    /// F2 offset (Hz)、母音空間の横軸微調整
    pub formant_f2_offset_hz: f32,
    /// Formant bandwidth scale、tense → 0.7、breathy → 1.3
    pub bandwidth_scale: f32,
    /// Spectral tilt (dB/octave)、bright → +2、dark → -3
    pub spectral_tilt_db: f32,
    /// Aspiration (breathiness) gain、[0.0, 1.0]
    pub breathiness: f32,
    /// Pitch jitter (%)、voice quality
    pub jitter_pct: f32,
    /// Amplitude shimmer (%)、voice quality
    pub shimmer_pct: f32,
}
// 10 × f32 = 40 bytes
```

### Preset examples

```rust
impl VoiceParams {
    pub const fn male_low() -> Self { Self {
        f0_mean_hz: 95.0, f0_range_scale: 0.8, formant_scale: 0.82,
        formant_f1_offset_hz: -30.0, formant_f2_offset_hz: -50.0,
        bandwidth_scale: 1.0, spectral_tilt_db: -2.0,
        breathiness: 0.1, jitter_pct: 0.5, shimmer_pct: 1.0,
    }}
    pub const fn female_high() -> Self { Self {
        f0_mean_hz: 240.0, f0_range_scale: 1.3, formant_scale: 1.05,
        formant_f1_offset_hz: 20.0, formant_f2_offset_hz: 40.0,
        bandwidth_scale: 0.95, spectral_tilt_db: 1.0,
        breathiness: 0.2, jitter_pct: 0.3, shimmer_pct: 0.8,
    }}
    pub const fn child_girl() -> Self { Self {
        f0_mean_hz: 280.0, f0_range_scale: 1.4, formant_scale: 1.15,
        formant_f1_offset_hz: 40.0, formant_f2_offset_hz: 80.0,
        bandwidth_scale: 1.0, spectral_tilt_db: 2.0,
        breathiness: 0.15, jitter_pct: 0.4, shimmer_pct: 1.2,
    }}
    pub const fn whisper() -> Self { Self {
        f0_mean_hz: 0.0, // unvoiced
        f0_range_scale: 0.0, formant_scale: 1.0,
        formant_f1_offset_hz: 0.0, formant_f2_offset_hz: 0.0,
        bandwidth_scale: 1.5, spectral_tilt_db: -4.0,
        breathiness: 0.9, jitter_pct: 2.0, shimmer_pct: 3.0,
    }}
    pub const fn robot() -> Self { Self {
        f0_mean_hz: 150.0, f0_range_scale: 0.1, // monotone
        formant_scale: 1.0,
        formant_f1_offset_hz: 0.0, formant_f2_offset_hz: 0.0,
        bandwidth_scale: 0.5, spectral_tilt_db: 0.0,
        breathiness: 0.0, jitter_pct: 0.0, shimmer_pct: 0.0,
    }}
    // ... more presets
}
```

## 3. EmotionParams schema (40 bytes)

```rust
/// 感情を parametric 表現する 40-byte struct
#[repr(C)]
pub struct EmotionParams {
    /// Pitch range multiplier (excitement)、calm → 0.7、excited → 1.5
    pub pitch_range_scale: f32,
    /// Pitch center shift (semitones)、happy → +2、sad → -3
    pub pitch_shift_semitones: f32,
    /// Tempo multiplier、angry → 1.15、sad → 0.85
    pub tempo_scale: f32,
    /// Energy multiplier、loud → 1.5、whisper → 0.3
    pub energy_scale: f32,
    /// Brightness (spectral emphasis)、happy → +2dB、sad → -3dB
    pub brightness_db: f32,
    /// Pitch volatility (angry → 1.8、calm → 0.5)
    pub pitch_volatility_scale: f32,
    /// Energy contour dynamics (sad → smooth、excited → punchy)
    pub energy_dynamics_scale: f32,
    /// Vibrato depth (%)、operatic → 8、pop → 3、speech → 0
    pub vibrato_depth_pct: f32,
    /// Vibrato rate (Hz)、typical 4-7
    pub vibrato_rate_hz: f32,
    /// Voice quality shift (tense → +1、smooth → -1)
    pub tension: f32,
}
// 10 × f32 = 40 bytes
```

### Preset examples

```rust
impl EmotionParams {
    pub const fn neutral() -> Self { /* all defaults */ }
    pub const fn happy() -> Self { Self {
        pitch_range_scale: 1.3, pitch_shift_semitones: 2.0,
        tempo_scale: 1.1, energy_scale: 1.2, brightness_db: 2.0,
        pitch_volatility_scale: 1.4, energy_dynamics_scale: 1.3,
        vibrato_depth_pct: 0.0, vibrato_rate_hz: 5.0, tension: 0.2,
    }}
    pub const fn sad() -> Self { Self {
        pitch_range_scale: 0.7, pitch_shift_semitones: -3.0,
        tempo_scale: 0.85, energy_scale: 0.8, brightness_db: -3.0,
        pitch_volatility_scale: 0.6, energy_dynamics_scale: 0.7,
        vibrato_depth_pct: 0.0, vibrato_rate_hz: 5.0, tension: -0.3,
    }}
    pub const fn angry() -> Self { Self {
        pitch_range_scale: 1.5, pitch_shift_semitones: 1.0,
        tempo_scale: 1.15, energy_scale: 1.5, brightness_db: 3.0,
        pitch_volatility_scale: 1.8, energy_dynamics_scale: 1.6,
        vibrato_depth_pct: 0.0, vibrato_rate_hz: 5.0, tension: 0.8,
    }}
    pub const fn calm() -> Self { Self {
        pitch_range_scale: 0.85, pitch_shift_semitones: -0.5,
        tempo_scale: 0.95, energy_scale: 0.9, brightness_db: -1.0,
        pitch_volatility_scale: 0.5, energy_dynamics_scale: 0.6,
        vibrato_depth_pct: 0.0, vibrato_rate_hz: 5.0, tension: -0.2,
    }}
    // ... more presets
}
```

## 4. Post-process math

### 4.1 Voice transform (`apply_voice_params`)

```rust
pub fn apply_voice_params(mel: &Mel, voice: &VoiceParams, sr: u32) -> Mel {
    let mut out = mel.clone();
    // 1. Formant frequency warping (mel freq axis piecewise linear warp)
    out = warp_frequency_axis(&out, voice.formant_scale);
    // 2. F1/F2 offset (fine adjustment)
    out = shift_formant(&out, voice.formant_f1_offset_hz, 1, sr);
    out = shift_formant(&out, voice.formant_f2_offset_hz, 2, sr);
    // 3. Bandwidth scaling (spectral peak sharpness)
    out = scale_bandwidth(&out, voice.bandwidth_scale);
    // 4. Spectral tilt (brightness)
    out = apply_spectral_tilt(&out, voice.spectral_tilt_db, sr);
    // 5. Breathiness (mix aspiration noise in high freq)
    out = add_breathiness(&out, voice.breathiness);
    out
}
```

**個別 math**:

- **`warp_frequency_axis(mel, scale)`**: mel[t, f] → mel[t, f × scale]、piecewise linear interpolation
- **`shift_formant(mel, hz_offset, formant_idx, sr)`**: local resonance peak を hz_offset だけ ずらす
- **`scale_bandwidth(mel, scale)`**: spectral envelope smoothing (scale > 1 で broader peaks)
- **`apply_spectral_tilt(mel, db_per_octave, sr)`**: log-frequency ramp を multiply
- **`add_breathiness(mel, gain)`**: gain × white_noise × high_freq_envelope を mel に加算

### 4.2 Emotion transform (`apply_emotion_params`)

```rust
pub fn apply_emotion_params(mel: &Mel, emotion: &EmotionParams, sr: u32, hop: usize) -> Mel {
    let mut out = mel.clone();
    // 1. Tempo scaling (time stretch)、pitch preservation
    out = time_stretch(&out, emotion.tempo_scale);
    // 2. F0 range/shift (pitch modulation)
    out = pitch_modulate(&out, emotion.pitch_range_scale,
                         emotion.pitch_shift_semitones,
                         emotion.pitch_volatility_scale);
    // 3. Energy envelope shaping
    out = shape_energy(&out, emotion.energy_scale, emotion.energy_dynamics_scale);
    // 4. Brightness EQ
    out = apply_spectral_tilt(&out, emotion.brightness_db, sr);
    // 5. Vibrato (F0 sinusoidal modulation)
    if emotion.vibrato_depth_pct > 0.0 {
        out = apply_vibrato(&out, emotion.vibrato_depth_pct,
                            emotion.vibrato_rate_hz, hop, sr);
    }
    // 6. Tension shift (voice quality)
    out = shift_tension(&out, emotion.tension);
    out
}
```

**個別 math** (representative):

- **`time_stretch(mel, scale)`**: mel[t] → mel[t / scale]、frames を re-sample (PSOLA-like on mel)
- **`pitch_modulate(mel, range_scale, shift_st, volatility)`**:
  - F0 contour 抽出 (autocorrelation or CREPE-lite)
  - `f0_new[t] = mean(f0) + (f0[t] - mean(f0)) × range_scale × volatility + shift`
  - Formant を維持しつつ harmonic 周波数だけ shift (WORLD 相当)
- **`shape_energy(mel, scale, dynamics)`**: envelope 抽出 → scale で乗算 + dynamics で contrast 強調
- **`apply_vibrato(mel, depth, rate, hop, sr)`**: `f0[t] × (1 + depth × sin(2π rate t))`

### 4.3 出力 pipeline

```rust
// 完全な synthesis pipeline
pub fn synthesize(
    base_model: &FastSpeech2,     // Layer 1
    voice: &VoiceParams,           // Layer 2
    emotion: &EmotionParams,       // Layer 3
    vocoder: &VocosModel,          // Layer 4
    mora_ids: &[u32],
    durations: &[u32],
    sr: u32,
    hop_length: usize,
) -> Vec<f32> {
    // Layer 1: neutral mel
    let mel_neutral = base_model.forward(mora_ids, 1, mora_ids.len(), durations).unwrap();

    // Layer 2: voice injection
    let mel_voiced = apply_voice_params(&mel_neutral, voice, sr);

    // Layer 3: emotion injection
    let mel_emotional = apply_emotion_params(&mel_voiced, emotion, sr, hop_length);

    // Layer 4: waveform synthesis
    vocoder.forward(&mel_emotional)
}
```

## 5. 現 FastSpeech2 実装との位置付け

**現在** (2026-07-30):
- ALICE-Train `src/tts/fastspeech2.rs` の FastSpeech2 model (hidden=128, layers=2)
- JSUT 5000 utt で学習中 (step ~15000 accumulated、loss ~2.3、~1MB weights)
- Griffin-Lim vocoder (テスト用)

**Layer 1 (base) として位置付け**:
- 現 model は「JSUT 女性話者に overfit した base」= **suboptimal Layer 1**
- 話者中立 (multi-speaker averaged or synthetic canonical) でなく、JSUT 個別声質を含む
- しかし **VoiceParams post-process で JSUT 女性 → 男性/子供/囁き 等に変換可能** (relative transformation)
- Phase A: 現 model を JSUT female base として使い、post-process で voice variety
- Phase C: 話者中立 base で再学習して transformation 精度上げる

## 6. Implementation phases

### Phase A: Post-process 追加 (追加学習不要、現 model そのまま活用)

**期間**: 1-2 週間
**成果物**:
- `src/tts/postprocess/voice.rs` — `VoiceParams` + `apply_voice_params()`
- `src/tts/postprocess/emotion.rs` — `EmotionParams` + `apply_emotion_params()`
- `src/tts/postprocess/presets.rs` — voice/emotion preset library
- `examples/fs2_synthesize_v2.rs` — voice + emotion post-process 込み合成
- Unit tests: math ops の invariance (formant scale 1.0 で恒等等)

**期待効果**:
- 現 model 1 個で 100+ voice × 20 emotion = 2000+ combination 出力可
- Payload 変わらず (~1MB base + 5KB presets)

### Phase B: Vocos integration + streaming API

**期間**: 1 週間
**成果物**:
- ALICE-TTS v1 の Vocos pretrained model を統合 (`src/tts/vocoder/vocos.rs`)
- Sentence-level streaming (`src/tts/stream/sentence.rs`)
- Real-time metric 計測 (first-audio latency、RTF)

**期待効果**:
- Griffin-Lim → Vocos で 20-100× 高速化 (~50ms first-audio)
- 会話 turn 単位 real-time 応答実現

### Phase C: Truly neutral base 再学習 (radical、追加学習必要)

**期間**: 3-4 週間 (data prep + train)
**成果物**:
- Multi-speaker corpus (JVS 100 話者) or synthetic canonical で base 再学習
- 話者性を model から抜き、VoiceParams への transformation 精度向上
- `src/tts/data/multispeaker.rs` — averaging / normalization

**期待効果**:
- VoiceParams による話者変換品質が natural に
- Base model 単体では「特徴なき中立声」、post-process で個性

### Phase D: Ternary QAT + edge deploy (ALICE-LLM 経験の応用)

**期間**: 2-3 週間
**成果物**:
- Base model の Ternary QAT (~1MB → ~200KB)
- Jetson Orin Nano / RPi 5 実機動作
- WebAssembly ビルド (browser 動作)

**期待効果**:
- Total payload: **200KB base + 5KB presets + Vocos ~1MB = ~1.2MB**
- ALICE-Font の 200KB radical さに近づく
- Edge device system voice (AI-Tencho, ALICE-Anima) に直接組込み可

## 7. Historical precedent (音声科学の decomposition 知見)

このアーキテクチャは音声科学の以下の知見の再統合:

- **Klatt synthesizer** (Klatt 1980): parametric formant synthesizer、40-parameter voice model
- **PSOLA** (Moulines & Charpentier 1990): Pitch-Synchronous Overlap-Add、pitch/duration 独立変更
- **VTLN** (Vocal Tract Length Normalization、1990s): formant warping による speaker adaptation
- **STRAIGHT** (Kawahara 1999): F0 / spectral envelope / aperiodicity 独立分離分析合成
- **WORLD vocoder** (Morise 2016): STRAIGHT の open-source 実装、高品質分析合成
- **HTS** (HMM-based Speech Synthesis, 2000s): 統計的 parametric synthesis
- **Neural WORLD**: Neural network で WORLD parameter 予測

**modern neural TTS (Tacotron / FastSpeech / VITS) はこれらを "忘れて" end-to-end mel prediction に統合した**
→ voice/emotion が model 内に bake され、変更が再学習必要になった
→ ALICE 原則違反 (data smuggling)

**ALICE-radical TTS は音声科学の decomposition 精神を neural + parametric hybrid で復活させる**。

## 8. ALICE 原則との対応表

| ALICE 原則 | ALICE-Font 実装 | ALICE-radical TTS 実装 |
|-----------|----------------|----------------------|
| Ship laws not data | Bezier stroke skeletons + IDS rules | Phoneme neural model + post-process math |
| Small payload | 200KB total | ~1MB base + 5KB presets |
| Parametric variation | MetaFontParams 40B → 無限 style | VoiceParams 40B × EmotionParams 40B → 無限 (voice × emotion) |
| Runtime reconstruction | fragment shader analytical Bezier | mel generation + math post-process + vocoder |
| No baked data | ~ (radicals は library だが parametric) | Base model は "baked learning" だが post-process 層で分離 |
| Resolution independent | ✓ (数学式、任意 zoom) | ⚠️ (sample rate 依存、mel frame rate 固定) |

**現時点で ALICE-Font には及ばないが、pragmatic learned 側で最大限 radical**。

## 9. Alternative: pure analytical TTS (超 radical)

**もし完全 analytical に振り切るなら** (Klatt synthesizer 系):

- Base model = **なし** (neural 一切なし)
- Phoneme library: 42 mora × formant trajectory (~5KB)
- Prosody model: 数式的 F0 curve model (accent rule + duration model、~5KB)
- Vocal tract filter: source-filter math (~2KB code)
- **Total: ~12KB** for a full voice
- **Quality**: 1980s Klatt synthesizer level (機械声、"HAL 9000")

**Trade-off**:
- **Font**: analytical で 90% 品質達成可 (bezier 完全表現)
- **TTS**: analytical で 40-50% 品質のみ (voice 統計的性質を数式で捕捉困難)
- → ALICE-Font の完全 radical と同精神は TTS では品質犠牲大
- → **Pragmatic learned base + parametric post-process が現実的着地**

**Reference implementation** (将来の investigation):
- `alice-tts-parametric` crate として別途起票
- Klatt synthesizer Rust 実装、~12KB voice、比較評価

## 10. Roadmap 統合

**ALICE-Train (current) との統合**:

- Phase A (post-process): `src/tts/postprocess/` module 追加、追加学習不要
- Phase B (Vocos): `src/tts/vocoder/vocos.rs` 追加、ALICE-TTS v1 参照
- Phase C (neutral base): 現学習完了後の再設計
- Phase D (Ternary QAT): ALICE-LLM の QAT パイプラインを TTS base に適用

**分離 crate 検討 (将来)**:

- `alice-tts-postprocess` — voice/emotion params + math ops、standalone crate
- `alice-tts-parametric` — Klatt synthesizer 実装、pure analytical baseline
- `alice-tts-vocos` — Vocos vocoder Rust binding

**成果物 target (最終)**:

```
Total: ~1.2MB payload → 無限 (voice × emotion × text) combination
├── alice-tts-base (Ternary QAT model)      ~200KB
├── alice-tts-postprocess (voice + emotion)  ~50KB (code + 100 presets)
└── alice-tts-vocos (fixed vocoder)          ~1MB
```

**ALICE-Font 200KB に対して TTS 1.2MB — voice complexity 分の overhead はあるが、
同 order of magnitude で ALICE 原則を体現できる**。

## 11. Open questions

1. **Base model は本当に voice-neutral 化可能か?**
   - Multi-speaker averaging で「平均声」は情報損失で不自然にならないか?
   - Synthetic canonical (Klatt で合成した中性声) を教師データにする方が良い?

2. **VoiceParams 40 byte で voice の bulk を捉えられるか?**
   - 話者性は formant + F0 + jitter/shimmer で捕捉可能?
   - 「〇〇さんの声」レベルの個体差は 40B に収まる?
   - VoiceParams を 80B / 200B に拡大した場合の quality curve?

3. **Emotion post-process の品質限界は?**
   - 「悲しみ」等の複雑感情は F0 range + tempo + energy だけで表現可能?
   - Voice quality (breathiness, tension) の変化と組合わせで足りる?
   - 感情ごとの subtle prosody pattern (e.g. 悲しみの sighing) は post-process では出せない?

4. **Vocos vocoder は voice/emotion post-process 後の mel を正しく音声化できるか?**
   - Vocos は learned mel → wave mapping、学習分布外の modified mel で崩壊する可能性
   - Robust vocoder 選定 or vocoder side でも parametric 拡張余地?

5. **Streaming mora-level post-process は可能か?**
   - Sentence-level batching vs mora-level online processing
   - Latency vs 音声連続性 trade-off

## 12. 参照

- ALICE-Font README: `~/ALICE-Font/README.md`
- ALICE-Font ROADMAP: `~/ALICE-Font/ROADMAP.md`
- ALICE-TTS current (FastSpeech2): `~/ALICE-Train/src/tts/`
- ALICE-TTS v1 (Vocos baseline): `~/ALICE-TTS/src/`
- feedback_checkpoint_naming_resume: `~/.claude/projects/-Users-ys/memory/feedback_checkpoint_naming_resume.md`

## 13. History

- 2026-07-30: draft-0.1 作成 (ALICE-Font principle 適用の設計 doc)
  - User 指摘「声質・感情は後付け可能」→ decomposition 前提
  - Phase A-D roadmap
  - Klatt-style pure analytical alternative の明記
