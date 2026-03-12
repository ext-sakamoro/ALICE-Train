//! 学習ログ — loss/lr/grad_norm の記録と出力。
//!
//! CSV/JSON 形式でログを保存し、学習の進捗を追跡可能にする。

use serde::Serialize;
use std::io;
use std::path::Path;

/// 1ステップの学習ログエントリ。
#[derive(Clone, Debug, Serialize)]
pub struct LogEntry {
    /// エポック番号。
    pub epoch: usize,
    /// ステップ番号（グローバル）。
    pub step: usize,
    /// 損失値。
    pub loss: f32,
    /// 学習率。
    pub learning_rate: f32,
    /// 勾配ノルム（L2）。0.0 = 未計算。
    pub grad_norm: f32,
}

impl LogEntry {
    /// 新しいログエントリを作成する。
    #[must_use]
    pub fn new(epoch: usize, step: usize, loss: f32, learning_rate: f32, grad_norm: f32) -> Self {
        Self {
            epoch,
            step,
            loss,
            learning_rate,
            grad_norm,
        }
    }
}

/// 学習ログ。
///
/// ログエントリを蓄積し、CSV/JSON に出力する。
#[derive(Clone, Debug)]
pub struct TrainLog {
    /// 蓄積されたログエントリ。
    entries: Vec<LogEntry>,
}

impl TrainLog {
    /// 空のログを作成する。
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// 指定容量で空のログを作成する。
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
        }
    }

    /// ログエントリを追加する。
    pub fn append(&mut self, entry: LogEntry) {
        self.entries.push(entry);
    }

    /// エントリ数を返す。
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// ログが空かどうか。
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 全エントリへの参照を返す。
    #[must_use]
    pub fn entries(&self) -> &[LogEntry] {
        &self.entries
    }

    /// 最後のエントリを返す。
    #[must_use]
    pub fn last(&self) -> Option<&LogEntry> {
        self.entries.last()
    }

    /// CSV 形式で書き出す。
    ///
    /// # Errors
    ///
    /// I/O エラー時。
    pub fn save_csv<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        writeln!(writer, "epoch,step,loss,learning_rate,grad_norm")?;
        for e in &self.entries {
            writeln!(
                writer,
                "{},{},{},{},{}",
                e.epoch, e.step, e.loss, e.learning_rate, e.grad_norm
            )?;
        }
        Ok(())
    }

    /// CSV ファイルに保存する。
    ///
    /// # Errors
    ///
    /// I/O エラー時。
    pub fn save_csv_to_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut writer = io::BufWriter::new(file);
        self.save_csv(&mut writer)
    }

    /// JSON 形式で書き出す。
    ///
    /// # Errors
    ///
    /// I/O エラー時。
    pub fn save_json<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        let json = serde_json::to_string_pretty(&self.entries)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        writer.write_all(json.as_bytes())
    }

    /// JSON ファイルに保存する。
    ///
    /// # Errors
    ///
    /// I/O エラー時。
    pub fn save_json_to_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut writer = io::BufWriter::new(file);
        self.save_json(&mut writer)
    }
}

impl Default for TrainLog {
    fn default() -> Self {
        Self::new()
    }
}

/// 勾配の L2 ノルムを計算する。
///
/// `grad_norm = sqrt(sum(g_i^2))`
#[must_use]
pub fn compute_grad_norm(gradients: &[f32]) -> f32 {
    let mut sum_sq = 0.0f64;
    for &g in gradients {
        sum_sq += (g as f64) * (g as f64);
    }
    (sum_sq.sqrt()) as f32
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // --- LogEntry ---

    #[test]
    fn log_entry_fields() {
        let e = LogEntry::new(1, 100, 0.5, 0.001, 1.23);
        assert_eq!(e.epoch, 1);
        assert_eq!(e.step, 100);
        assert!((e.loss - 0.5).abs() < 1e-6);
        assert!((e.learning_rate - 0.001).abs() < 1e-6);
        assert!((e.grad_norm - 1.23).abs() < 1e-6);
    }

    #[test]
    fn log_entry_clone() {
        let e = LogEntry::new(0, 0, 1.0, 0.01, 0.0);
        let e2 = e.clone();
        assert_eq!(e2.epoch, 0);
    }

    #[test]
    fn log_entry_debug() {
        let e = LogEntry::new(0, 0, 1.0, 0.01, 0.0);
        let s = format!("{e:?}");
        assert!(s.contains("LogEntry"));
    }

    // --- TrainLog ---

    #[test]
    fn train_log_new_empty() {
        let log = TrainLog::new();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
        assert!(log.last().is_none());
    }

    #[test]
    fn train_log_default() {
        let log = TrainLog::default();
        assert!(log.is_empty());
    }

    #[test]
    fn train_log_append_and_len() {
        let mut log = TrainLog::new();
        log.append(LogEntry::new(0, 0, 1.0, 0.01, 0.0));
        log.append(LogEntry::new(0, 1, 0.9, 0.01, 0.5));
        assert_eq!(log.len(), 2);
        assert!(!log.is_empty());
    }

    #[test]
    fn train_log_last() {
        let mut log = TrainLog::new();
        log.append(LogEntry::new(0, 0, 1.0, 0.01, 0.0));
        log.append(LogEntry::new(1, 10, 0.5, 0.001, 0.3));
        let last = log.last().unwrap();
        assert_eq!(last.epoch, 1);
        assert!((last.loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn train_log_entries() {
        let mut log = TrainLog::new();
        log.append(LogEntry::new(0, 0, 1.0, 0.01, 0.0));
        let entries = log.entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].step, 0);
    }

    #[test]
    fn train_log_with_capacity() {
        let log = TrainLog::with_capacity(100);
        assert!(log.is_empty());
    }

    #[test]
    fn train_log_save_csv() {
        let mut log = TrainLog::new();
        log.append(LogEntry::new(0, 0, 1.5, 0.01, 0.1));
        log.append(LogEntry::new(0, 1, 1.2, 0.01, 0.2));

        let mut buf = Vec::new();
        log.save_csv(&mut buf).unwrap();
        let csv = String::from_utf8(buf).unwrap();

        assert!(csv.starts_with("epoch,step,loss,learning_rate,grad_norm\n"));
        assert!(csv.contains("0,0,1.5,0.01,0.1"));
        assert!(csv.contains("0,1,1.2,0.01,0.2"));
    }

    #[test]
    fn train_log_save_csv_empty() {
        let log = TrainLog::new();
        let mut buf = Vec::new();
        log.save_csv(&mut buf).unwrap();
        let csv = String::from_utf8(buf).unwrap();
        assert_eq!(csv.lines().count(), 1); // header only
    }

    #[test]
    fn train_log_save_json() {
        let mut log = TrainLog::new();
        log.append(LogEntry::new(0, 0, 1.0, 0.01, 0.0));

        let mut buf = Vec::new();
        log.save_json(&mut buf).unwrap();
        let json = String::from_utf8(buf).unwrap();

        assert!(json.contains("\"epoch\": 0"));
        assert!(json.contains("\"loss\""));
    }

    #[test]
    fn train_log_save_csv_to_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log.csv");

        let mut log = TrainLog::new();
        log.append(LogEntry::new(0, 0, 1.0, 0.01, 0.0));
        log.save_csv_to_file(&path).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("epoch,step"));
    }

    #[test]
    fn train_log_save_json_to_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log.json");

        let mut log = TrainLog::new();
        log.append(LogEntry::new(0, 0, 1.0, 0.01, 0.0));
        log.save_json_to_file(&path).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("\"epoch\""));
    }

    #[test]
    fn train_log_clone() {
        let mut log = TrainLog::new();
        log.append(LogEntry::new(0, 0, 1.0, 0.01, 0.0));
        let log2 = log.clone();
        assert_eq!(log2.len(), 1);
    }

    #[test]
    fn train_log_debug() {
        let log = TrainLog::new();
        let s = format!("{log:?}");
        assert!(s.contains("TrainLog"));
    }

    // --- compute_grad_norm ---

    #[test]
    fn grad_norm_zero() {
        assert!((compute_grad_norm(&[0.0, 0.0, 0.0]) - 0.0).abs() < 1e-8);
    }

    #[test]
    fn grad_norm_unit() {
        assert!((compute_grad_norm(&[1.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn grad_norm_3_4_5() {
        // sqrt(3^2 + 4^2) = 5
        assert!((compute_grad_norm(&[3.0, 4.0]) - 5.0).abs() < 1e-4);
    }

    #[test]
    fn grad_norm_negative() {
        assert!((compute_grad_norm(&[-3.0, 4.0]) - 5.0).abs() < 1e-4);
    }

    #[test]
    fn grad_norm_empty() {
        assert!((compute_grad_norm(&[]) - 0.0).abs() < 1e-8);
    }

    #[test]
    fn grad_norm_large() {
        let grads: Vec<f32> = (0..1000).map(|i| i as f32 * 0.01).collect();
        let norm = compute_grad_norm(&grads);
        assert!(norm.is_finite());
        assert!(norm > 0.0);
    }
}
