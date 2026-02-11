// LLM模块 - 管理AI分析服务

pub mod claude;
pub mod codex;
pub mod plugin;
pub mod qwen;
pub mod ollama;
pub use ollama::OllamaProvider;


pub use claude::ClaudeProvider;
pub use codex::CodexProvider;
pub use plugin::{
    ActivityCategory, ActivityTag, AppSites, Distraction, KeyMoment, LLMProvider, SessionBrief,
    SessionSummary, TimelineCard, VideoSegment,
};
pub use qwen::QwenProvider;

use crate::capture::scheduler::SessionProcessor;
use crate::settings::SettingsManager;
use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

/// LLM管理器
pub struct LLMManager {
    /// 当前使用的提供商
    provider: Box<dyn LLMProvider>,
    /// 配置锁
    config_lock: Arc<RwLock<LLMConfig>>,
    /// HTTP 客户端（用于 Qwen provider）
    http_client: Option<reqwest::Client>,
}

/// LLM配置
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LLMConfig {
    /// 当前使用的 provider: "qwen" 或 "claude"
    #[serde(default = "default_provider")]
    pub provider: String,
    /// Qwen配置
    pub qwen: QwenConfig,
    /// Claude配置（目前无需额外配置）
    #[serde(default)]
    pub claude: ClaudeConfig,
    /// Ollama 配置
    #[serde(default)]
    pub ollama: OllamaConfig,
    /// Codex配置
    #[serde(default)]
    pub codex: CodexConfig,
    /// 分析参数
    pub analysis_params: AnalysisParams,
}

fn default_provider() -> String {
    "qwen".to_string()
}

/// Claude配置（使用 Agent SDK，可通过配置传入 API key）
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct ClaudeConfig {
    pub api_key: Option<String>,
    pub model: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OllamaConfig {
    #[serde(default = "default_ollama_base_url")]
    pub base_url: String,
    #[serde(default = "default_ollama_model")]
    pub model: String,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: default_ollama_base_url(),
            model: default_ollama_model(),
        }
    }
}

fn default_ollama_base_url() -> String {
    "http://100.82.18.91:11434".to_string()
}

fn default_ollama_model() -> String {
    "qwen3-vl:32b".to_string()
}

/// Codex配置
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct CodexConfig {
    #[serde(default)]
    pub binary_path: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub profile: Option<String>,
    #[serde(default)]
    pub sandbox_mode: Option<String>,
    #[serde(default)]
    pub approval_policy: Option<String>,
    #[serde(default)]
    pub extra_args: Vec<String>,
    #[serde(default)]
    pub max_images: Option<usize>,
    #[serde(default)]
    pub timeout_secs: Option<u64>,
    #[serde(default)]
    pub summary_prompt: Option<String>,
    #[serde(default)]
    pub segment_prompt: Option<String>,
    #[serde(default)]
    pub timeline_prompt: Option<String>,
    #[serde(default)]
    pub day_summary_prompt: Option<String>,
}

/// Qwen配置
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QwenConfig {
    pub api_key: String,
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default = "default_base_url")]
    pub base_url: String,
    #[serde(default = "default_video_mode")]
    pub use_video_mode: bool, // 是否使用视频模式处理多张图片
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video_path: Option<String>, // 可选的视频文件路径
}

fn default_model() -> String {
    "qwen-vl-max-latest".to_string()
}

fn default_base_url() -> String {
    "https://dashscope.aliyuncs.com/compatible-mode/v1".to_string()
}

fn default_video_mode() -> bool {
    true
}

/// 分析参数
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AnalysisParams {
    /// 帧采样间隔（秒）
    pub frame_sampling_interval: u64,
    /// 最大分析帧数
    pub max_frames_per_analysis: usize,
    /// 是否包含详细描述
    pub include_detailed_description: bool,
    /// 置信度阈值
    pub confidence_threshold: f32,
}

impl Default for AnalysisParams {
    fn default() -> Self {
        Self {
            frame_sampling_interval: 30, // 每30秒采样一帧
            max_frames_per_analysis: 30, // 最多30帧
            include_detailed_description: true,
            confidence_threshold: 0.5,
        }
    }
}

impl LLMManager {
    /// 创建新的LLM管理器（接受共享的HTTP客户端以复用连接池）
    pub fn new(client: reqwest::Client) -> Self {
        // 默认使用 Qwen provider
        let provider: Box<dyn LLMProvider> = Box::new(QwenProvider::new(client.clone()));

        Self {
            provider,
            config_lock: Arc::new(RwLock::new(LLMConfig {
                provider: default_provider(),
                qwen: QwenConfig {
                    api_key: String::new(),
                    model: default_model(),
                    base_url: default_base_url(),
                    use_video_mode: default_video_mode(),
                    video_path: None,
                },
                claude: ClaudeConfig::default(),
                codex: CodexConfig::default(),
                ollama: OllamaConfig::default(),
                analysis_params: AnalysisParams::default(),
            })),
            http_client: Some(client),
        }
    }

    /// 配置 LLM（支持多 provider）
    pub async fn configure(&mut self, config: QwenConfig) -> Result<()> {
        // 获取当前 provider 类型
        let current_config = self.config_lock.read().await;
        let provider_name = current_config.provider.clone();
        drop(current_config);

        match provider_name.as_str() {
            "qwen" | "openai" => {
                // "openai" 是 Qwen 的别名（因为使用 OpenAI 兼容接口）
                info!("配置 Qwen provider (当前provider名称: {})", provider_name);

                // 如果有视频路径，设置到provider
                if let Some(ref video_path) = config.video_path {
                    if let Some(provider) = self.provider.as_any().downcast_mut::<QwenProvider>() {
                        provider.set_video_path(Some(video_path.clone()));
                    }
                }

                // 更新provider配置
                let config_json = serde_json::to_value(&config)?;
                info!("应用 Qwen 配置: {:?}", config_json);
                self.provider.configure(config_json)?;

                // 更新配置锁
                let mut current_config = self.config_lock.write().await;
                current_config.qwen = config;

                info!("Qwen 配置已更新成功");
            }
            "claude" => {
                // Claude 目前无需额外配置
                info!("Claude 配置已更新（无需额外配置）");
            }
            _ => {
                warn!("未知的 provider: {}", provider_name);
            }

        }

        Ok(())
    }

    /// 切换 provider（不需要 client 参数，使用内部保存的 client）
    pub async fn switch_provider(&mut self, provider_name: &str) -> Result<()> {
        // 检查是否已经是目标 provider，避免重复创建实例
        let current_config = self.config_lock.read().await;
        let current_provider = current_config.provider.clone();
        drop(current_config);

        if current_provider == provider_name {
            info!("Provider 已经是 {}，无需切换", provider_name);
            return Ok(());
        }

        info!(
            "切换 LLM provider: {} -> {}",
            current_provider, provider_name
        );

        match provider_name {
            "qwen" | "openai" => {
                // 需要 HTTP 客户端
                let client = self
                    .http_client
                    .clone()
                    .ok_or_else(|| anyhow!("无法切换到 Qwen provider: HTTP 客户端未初始化"))?;
                self.provider = Box::new(QwenProvider::new(client));
            }
            "claude" => {
                self.provider = Box::new(ClaudeProvider::new());
            }
            "codex" => {
                self.provider = Box::new(CodexProvider::new());
            }
            // "ollama" => {
            //     self.provider = Box::new(OllamaProvider::new());
            // }
            "ollama" => {
                let client = self
                    .http_client
                    .clone()
                    .ok_or_else(|| anyhow!("无法切换到 Ollama provider: HTTP 客户端未初始化"))?;
                self.provider = Box::new(OllamaProvider::new(client));

                // 可选：应用已保存的 ollama 配置
                let cfg = { self.config_lock.read().await.ollama.clone() };
                self.provider.configure(serde_json::to_value(cfg)?)?;
            }
            _ => {
                return Err(anyhow!("不支持的 provider: {}", provider_name));
            }
        }

        // 更新配置中的 provider
        let mut config = self.config_lock.write().await;
        config.provider = provider_name.to_string();

        info!("已切换到 provider: {}", provider_name);
        Ok(())
    }

    /// 配置 Claude provider
    pub async fn configure_claude(&mut self, config: serde_json::Value) -> Result<()> {
        info!("配置 Claude provider");

        // 更新 provider 配置
        self.provider.configure(config.clone())?;

        // 更新配置锁中的 Claude 配置
        let mut current_config = self.config_lock.write().await;
        if let Some(api_key) = config.get("api_key").and_then(|v| v.as_str()) {
            current_config.claude.api_key = Some(api_key.to_string());
        }
        if let Some(model) = config.get("model").and_then(|v| v.as_str()) {
            current_config.claude.model = Some(model.to_string());
        }

        info!("Claude 配置已更新");
        Ok(())
    }
    /// 配置 ollama provider
    pub async fn configure_ollama(&mut self, config: OllamaConfig) -> Result<()> {
        // 如果当前 provider 不是 ollama，也允许配置（只是更新 config 存储）
        let cfg_json = serde_json::to_value(&config)?;
        if let Some(p) = self.provider.as_any().downcast_mut::<OllamaProvider>() {
            p.configure(cfg_json)?;
        } else {
            // 如果当前 provider 不是 ollama，但你想切换过去：
            self.provider = Box::new(OllamaProvider::new(
                self.http_client.clone().ok_or_else(|| anyhow!("HTTP client missing"))?,
            ));
            self.provider.configure(cfg_json)?;
        }

        let mut current = self.config_lock.write().await;
        current.provider = "ollama".to_string();
        current.ollama = config;
        Ok(())
    }

    /// 配置 Codex provider
    pub async fn configure_codex(&mut self, config: CodexConfig) -> Result<()> {
        info!("配置 Codex provider");

        let config_value = serde_json::to_value(&config)?;

        if let Some(provider) = self.provider.as_any().downcast_mut::<CodexProvider>() {
            provider.configure(config_value.clone())?;
        } else {
            warn!("当前 provider 不是 Codex，暂存配置待切换后生效");
        }

        let mut current_config = self.config_lock.write().await;
        current_config.codex = config;

        info!("Codex 配置已更新");
        Ok(())
    }

    pub fn set_video_path(&mut self, video_path: Option<String>) {
        if let Some(provider) = self.provider.as_any().downcast_mut::<QwenProvider>() {
            provider.set_video_path(video_path.clone());
        }

        if let Some(provider) = self.provider.as_any().downcast_mut::<ClaudeProvider>() {
            provider.set_video_path(video_path);
        }
    }

    /// 设置视频速率乘数
    pub fn set_video_speed(&mut self, speed_multiplier: f32) {
        // 只有 Qwen provider 需要视频速率
        if let Some(provider) = self.provider.as_any().downcast_mut::<QwenProvider>() {
            provider.set_video_speed(speed_multiplier);
        }
    }

    /// 设置会话时间范围（用于提示词中的绝对时间）
    pub fn set_session_window(&mut self, start: Option<DateTime<Utc>>, end: Option<DateTime<Utc>>) {
        self.provider.set_session_window(start, end);
    }

    /// 分析帧数据
    pub async fn analyze_frames(&mut self, frames: Vec<String>) -> Result<SessionSummary> {
        let provider_name = {
            let config = self.config_lock.read().await;
            config.provider.clone()
        };
        info!("使用 {} 分析 {} 帧", provider_name, frames.len());

        match self.provider.analyze_frames(frames).await {
            Ok(summary) => {
                info!("分析成功: {}", summary.title);
                Ok(summary)
            }
            Err(e) => {
                error!("分析失败: {}", e);
                Err(e)
            }
        }
    }

    /// 更新配置
    pub async fn update_config(&self, config: LLMConfig) -> Result<()> {
        let mut current_config = self.config_lock.write().await;
        *current_config = config;
        info!("LLM配置已更新");
        Ok(())
    }

    /// 获取当前配置
    pub async fn get_config(&self) -> LLMConfig {
        self.config_lock.read().await.clone()
    }

    /// 设置provider的数据库连接
    pub fn set_provider_database(
        &mut self,
        db: Arc<crate::storage::Database>,
        session_id: Option<i64>,
    ) {
        // Qwen provider
        if let Some(provider) = self.provider.as_any().downcast_mut::<QwenProvider>() {
            provider.set_database(db.clone());
            if let Some(sid) = session_id {
                provider.set_session_id(sid);
            }
            info!("为 Qwen provider 设置数据库连接");
            return;
        }

        // Claude provider
        if let Some(provider) = self.provider.as_any().downcast_mut::<ClaudeProvider>() {
            provider.set_database(db.clone());
            if let Some(sid) = session_id {
                provider.set_session_id(sid);
            }
            info!("为 Claude provider 设置数据库连接");
            return;
        }
        // Codex provider
        if let Some(provider) = self.provider.as_any().downcast_mut::<CodexProvider>() {
            provider.set_database(db.clone());
            if let Some(sid) = session_id {
                provider.set_session_id(sid);
            }
            info!("已为 Codex provider 设置数据库连接");
            return;
        }
        // Ollama provider
        if let Some(provider) = self.provider.as_any().downcast_mut::<OllamaProvider>() {
            provider.set_database(db.clone());
            if let Some(sid) = session_id {
                provider.set_session_id(sid);
            }
            info!("为 Ollama provider 设置数据库连接");
            return;
        }


    }

    /// 生成时间线卡片（公开方法）
    pub async fn generate_timeline(
        &mut self,
        segments: Vec<VideoSegment>,
        previous_cards: Option<Vec<TimelineCard>>,
    ) -> Result<Vec<TimelineCard>> {
        self.provider
            .generate_timeline(segments, previous_cards)
            .await
    }

    /// 获取最后一次LLM调用的ID
    pub fn get_last_call_id(&self, call_type: &str) -> Option<i64> {
        self.provider.last_llm_call_id(call_type)
    }

    /// 生成每日总结（调用LLM）
    pub async fn generate_day_summary(
        &self,
        date: &str,
        sessions: &[SessionBrief],
    ) -> Result<String> {
        self.provider.generate_day_summary(date, sessions).await
    }

    /// 分析视频并生成时间线（两阶段处理）
    pub async fn segment_video_and_generate_timeline(
        &mut self,
        frames: Vec<String>,
        duration: u32,
        previous_cards: Option<Vec<TimelineCard>>,
    ) -> Result<TimelineAnalysis> {
        let provider_name = {
            let config = self.config_lock.read().await;
            config.provider.clone()
        };

        info!(
            "使用 {} 进行视频分段分析: {} 帧, 时长 {} 分钟",
            provider_name,
            frames.len(),
            duration
        );

        // 第一阶段：分段视频
        let segments = match self.provider.segment_video(frames.clone(), duration).await {
            Ok(segs) => {
                info!("视频分段成功: {} 个segment", segs.len());
                segs
            }
            Err(e) => {
                error!("视频分段失败: {}", e);
                // 检查是否是视频过短错误，如果是则保留特殊标记
                if e.to_string().contains("VIDEO_TOO_SHORT") {
                    return Err(anyhow::anyhow!("VIDEO_TOO_SHORT: {}", e));
                }
                return Err(e);
            }
        };

        // 第二阶段：生成时间线
        let timeline_cards = match self
            .provider
            .generate_timeline(segments.clone(), previous_cards)
            .await
        {
            Ok(cards) => {
                info!("时间线生成成功: {} 个卡片", cards.len());
                cards
            }
            Err(e) => {
                error!("时间线生成失败: {}", e);
                return Err(e);
            }
        };

        let segment_call_id = self.provider.last_llm_call_id("segment_video");
        let timeline_call_id = self.provider.last_llm_call_id("generate_timeline");

        Ok(TimelineAnalysis {
            segments,
            timeline_cards,
            segment_call_id,
            timeline_call_id,
        })
    }
}

pub fn build_session_summary(
    window_start: chrono::DateTime<chrono::Utc>,
    window_end: chrono::DateTime<chrono::Utc>,
    segments: &[VideoSegment],
    timeline_cards: &[TimelineCard],
) -> SessionSummary {
    use std::collections::HashMap;

    // 统计各个类别的时间占比
    let mut category_duration: HashMap<String, f32> = HashMap::new();
    let mut total_duration = 0.0f32;

    // 遍历所有timeline cards计算时间
    for card in timeline_cards {
        // 解析时间并计算持续时间（简化处理，假设MM:SS格式）
        let duration = parse_duration(&card.start_time, &card.end_time).unwrap_or(15.0);
        total_duration += duration;

        let category = card.category.to_lowercase();
        *category_duration.entry(category).or_insert(0.0) += duration;
    }

    // 生成标签列表（最多3个，按比重排序）
    let mut tags: Vec<ActivityTag> = Vec::new();

    if total_duration > 0.0 {
        // 计算每个类别的比重并排序
        let mut category_weights: Vec<(String, f32)> = category_duration
            .into_iter()
            .map(|(cat, duration)| (cat, duration / total_duration))
            .collect();

        category_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // 取前3个类别生成标签
        for (category_str, weight) in category_weights.iter().take(3) {
            // 只添加比重超过10%的标签
            if *weight < 0.1 {
                break;
            }

            let category = match category_str.as_str() {
                "work" | "coding" | "writing" | "design" | "planning" | "data_analysis" => {
                    ActivityCategory::Work
                }
                "communication" | "meeting" => ActivityCategory::Communication,
                "learning" | "research" => ActivityCategory::Learning,
                "personal" | "entertainment" | "social_media" | "shopping" | "finance" => {
                    ActivityCategory::Personal
                }
                "idle" => ActivityCategory::Idle,
                _ => ActivityCategory::Other,
            };

            // 根据类别收集关键词
            let keywords = timeline_cards
                .iter()
                .filter(|card| card.category.to_lowercase() == *category_str)
                .map(|card| card.subcategory.clone())
                .collect::<Vec<_>>();

            tags.push(ActivityTag {
                category,
                confidence: *weight, // 使用时间占比作为confidence
                keywords,
            });
        }
    }

    // 如果没有有效标签，至少添加一个默认标签
    if tags.is_empty() && !timeline_cards.is_empty() {
        let first_card = &timeline_cards[0];
        tags.push(ActivityTag {
            category: map_category(&first_card.category),
            confidence: 1.0,
            keywords: vec![first_card.subcategory.clone()],
        });
    }

    // 生成总结
    let title = timeline_cards
        .first()
        .map(|c| c.title.clone())
        .unwrap_or_else(|| "活动会话".to_string());

    let summary = if timeline_cards.len() > 1 {
        format!(
            "本次会话包含{}个主要活动阶段。{}",
            timeline_cards.len(),
            timeline_cards
                .iter()
                .map(|c| c.title.clone())
                .collect::<Vec<_>>()
                .join("、")
        )
    } else {
        timeline_cards
            .first()
            .map(|c| {
                // 将 detailed_summary 中的相对时间转换为绝对时间
                convert_relative_times_in_text(&c.detailed_summary, window_start)
            })
            .unwrap_or_else(|| {
                segments
                    .first()
                    .map(|s| {
                        // 将 description 中的相对时间转换为绝对时间
                        convert_relative_times_in_text(&s.description, window_start)
                    })
                    .unwrap_or_default()
            })
    };

    SessionSummary {
        title,
        summary,
        tags,
        start_time: window_start,
        end_time: window_end,
        key_moments: segments
            .iter()
            .map(|seg| KeyMoment {
                time: seg.start_timestamp.clone(),
                description: seg.description.clone(),
                importance: 3,
            })
            .collect(),
        productivity_score: Some(75.0),
        focus_score: Some(80.0),
    }
}

// 辅助函数：解析持续时间（MM:SS格式）
fn parse_duration(start: &str, end: &str) -> Option<f32> {
    // 简单处理MM:SS格式
    let parse_time = |s: &str| -> Option<f32> {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() == 2 {
            let minutes = parts[0].parse::<f32>().ok()?;
            let seconds = parts[1].parse::<f32>().ok()?;
            Some(minutes * 60.0 + seconds)
        } else {
            None
        }
    };

    let start_seconds = parse_time(start)?;
    let end_seconds = parse_time(end)?;

    Some((end_seconds - start_seconds).abs())
}

// 辅助函数：映射类别
fn map_category(category_str: &str) -> ActivityCategory {
    match category_str.to_lowercase().as_str() {
        "work" | "coding" | "writing" | "design" | "planning" | "data_analysis" => {
            ActivityCategory::Work
        }
        "communication" | "meeting" => ActivityCategory::Communication,
        "learning" | "research" => ActivityCategory::Learning,
        "personal" | "entertainment" | "social_media" | "shopping" | "finance" => {
            ActivityCategory::Personal
        }
        "idle" => ActivityCategory::Idle,
        _ => ActivityCategory::Other,
    }
}

fn parse_relative_duration(label: &str) -> Option<chrono::Duration> {
    let trimmed = label.trim();
    if trimmed.is_empty() {
        return Some(chrono::Duration::zero());
    }

    let mut total_seconds: i64 = 0;
    let mut multiplier: i64 = 1;

    for part in trimmed.split(':').rev() {
        let component = part.trim();
        if component.is_empty() {
            continue;
        }
        let integer_part = component.split('.').next().unwrap_or(component).trim();
        let value = integer_part.parse::<i64>().ok()?;
        total_seconds = total_seconds.saturating_add(value.saturating_mul(multiplier));
        multiplier = multiplier.saturating_mul(60);
    }

    Some(chrono::Duration::seconds(total_seconds))
}

pub(crate) fn relative_to_absolute(
    start: chrono::DateTime<chrono::Utc>,
    end: chrono::DateTime<chrono::Utc>,
    relative: &str,
) -> chrono::DateTime<chrono::Utc> {
    let offset = parse_relative_duration(relative).unwrap_or_else(|| chrono::Duration::zero());
    let mut ts = start + offset;
    if ts > end {
        ts = end;
    }
    if ts < start {
        ts = start;
    }
    ts
}

/// 将文本中的相对时间（MM:SS 或 HH:MM:SS）转换为绝对时间（HH:MM）
fn convert_relative_times_in_text(
    text: &str,
    window_start: chrono::DateTime<chrono::Utc>,
) -> String {
    use chrono::Timelike;
    use regex::Regex;
    use std::sync::OnceLock;

    // 使用 OnceLock 缓存正则表达式对象
    static TIME_PATTERN: OnceLock<Regex> = OnceLock::new();
    let re = TIME_PATTERN.get_or_init(|| {
        // 匹配时间模式：MM:SS 或 HH:MM:SS
        // \b 确保是单词边界，避免匹配到更长的数字序列
        Regex::new(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b").unwrap()
    });

    let result = re.replace_all(text, |caps: &regex::Captures| {
        let time_str = caps.get(0).unwrap().as_str();

        // 转换为绝对时间
        let absolute_time = relative_to_absolute(
            window_start,
            window_start + chrono::Duration::days(1), // 使用一个足够大的结束时间
            time_str,
        );

        // 注意：window_start 虽然类型是 DateTime<Utc>，但实际值是本地时间
        // absolute_time 也是同样的情况，直接格式化即可，不要进行时区转换
        format!("{:02}:{:02}", absolute_time.hour(), absolute_time.minute())
    });

    result.to_string()
}

/// LLM处理器（实现SessionProcessor trait）
pub struct LLMProcessor {
    llm_handle: crate::actors::LLMHandle,
    db: Arc<crate::storage::Database>,
    video_processor: Option<Arc<crate::video::VideoProcessor>>,
    settings: Arc<SettingsManager>,
    notion_manager: Option<Arc<crate::notion::NotionManager>>,
}

/// LLM两阶段分析的聚合结果
pub struct TimelineAnalysis {
    pub segments: Vec<VideoSegment>,
    pub timeline_cards: Vec<TimelineCard>,
    pub segment_call_id: Option<i64>,
    pub timeline_call_id: Option<i64>,
}

impl LLMProcessor {
    pub fn new(
        llm_handle: crate::actors::LLMHandle,
        db: Arc<crate::storage::Database>,
        settings: Arc<SettingsManager>,
    ) -> Self {
        Self {
            llm_handle,
            db,
            video_processor: None,
            settings,
            notion_manager: None,
        }
    }

    pub fn with_video_processor(
        llm_handle: crate::actors::LLMHandle,
        db: Arc<crate::storage::Database>,
        video_processor: Arc<crate::video::VideoProcessor>,
        settings: Arc<SettingsManager>,
    ) -> Self {
        Self {
            llm_handle,
            db,
            video_processor: Some(video_processor),
            settings,
            notion_manager: None,
        }
    }

    /// 创建带视频处理器和 Notion 管理器的 LLMProcessor
    pub fn with_video_and_notion(
        llm_handle: crate::actors::LLMHandle,
        db: Arc<crate::storage::Database>,
        video_processor: Arc<crate::video::VideoProcessor>,
        settings: Arc<SettingsManager>,
        notion_manager: Arc<crate::notion::NotionManager>,
    ) -> Self {
        Self {
            llm_handle,
            db,
            video_processor: Some(video_processor),
            settings,
            notion_manager: Some(notion_manager),
        }
    }

    /// 启动事件监听器 - 监听SessionCompleted事件并执行分析
    pub async fn start_event_listener(
        self: Arc<Self>,
        event_bus: Arc<crate::event_bus::EventBus>,
        capture: Arc<crate::capture::ScreenCapture>,
    ) {
        let mut receiver = event_bus.subscribe();

        tokio::spawn(async move {
            info!("LLM处理器事件监听器已启动");

            while let Ok(event) = receiver.recv().await {
                match event {
                    crate::event_bus::AppEvent::SessionCompleted {
                        session_id,
                        frame_count,
                        window_start,
                        window_end,
                    } => {
                        info!(
                            "收到会话完成事件: session_id={}, frames={}, 时间段: {} - {}",
                            session_id, frame_count, window_start, window_end
                        );

                        // 发布分析开始事件
                        event_bus
                            .publish(crate::event_bus::AppEvent::AnalysisStarted { session_id });

                        // 读取该时间段的所有frames
                        let frames_result = Self::load_frames_for_window(
                            &capture,
                            session_id,
                            window_start,
                            window_end,
                        )
                        .await;

                        let frames = match frames_result {
                            Ok(f) => f,
                            Err(e) => {
                                error!("读取frames失败: {}", e);
                                event_bus.publish(crate::event_bus::AppEvent::AnalysisFailed {
                                    session_id,
                                    error: e.to_string(),
                                });
                                continue;
                            }
                        };

                        if frames.is_empty() {
                            warn!("该时间段没有有效frames，跳过分析");
                            event_bus.publish(crate::event_bus::AppEvent::AnalysisFailed {
                                session_id,
                                error: "没有有效frames".to_string(),
                            });
                            continue;
                        }

                        // 构建SessionWindow
                        let window = crate::capture::scheduler::SessionWindow {
                            start: window_start,
                            end: window_end,
                        };

                        // 执行分析
                        match self.process_session(frames, window).await {
                            Ok(_) => {
                                info!("会话分析完成: session_id={}", session_id);
                                // 注意：AnalysisCompleted事件将在未来由独立的分析流程发布
                                // 当前process_session包含了完整的处理，包括视频生成
                                // 这里暂时不发布AnalysisCompleted，避免重复处理
                            }
                            Err(e) => {
                                error!("会话分析失败: session_id={}, 错误: {}", session_id, e);
                                event_bus.publish(crate::event_bus::AppEvent::AnalysisFailed {
                                    session_id,
                                    error: e.to_string(),
                                });
                            }
                        }
                    }
                    _ => {}
                }
            }

            warn!("LLM处理器事件监听器已停止");
        });
    }

    /// 根据时间窗口加载frames
    async fn load_frames_for_window(
        capture: &Arc<crate::capture::ScreenCapture>,
        session_id: i64,
        window_start: chrono::DateTime<chrono::Utc>,
        window_end: chrono::DateTime<chrono::Utc>,
    ) -> Result<Vec<crate::capture::ScreenFrame>> {
        use chrono::TimeZone;

        let frames_dir = capture.frames_dir();
        if !frames_dir.exists() {
            return Err(anyhow!("frames目录不存在"));
        }

        let mut frames = Vec::new();
        let mut entries = tokio::fs::read_dir(&frames_dir).await?;

        let start_ms = window_start.timestamp_millis();
        let end_ms = window_end.timestamp_millis();

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();

            if !path.is_file() {
                continue;
            }

            let extension = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");
            if !extension.eq_ignore_ascii_case("jpg") {
                continue;
            }

            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };

            let Ok(timestamp_ms) = stem.parse::<i64>() else {
                continue;
            };

            // 检查是否在时间窗口内
            if timestamp_ms >= start_ms && timestamp_ms < end_ms {
                let Some(timestamp) = chrono::Utc.timestamp_millis_opt(timestamp_ms).single()
                else {
                    continue;
                };

                frames.push(crate::capture::ScreenFrame {
                    timestamp,
                    file_path: path.to_string_lossy().to_string(),
                    screen_id: 0,
                });
            }
        }

        // 按时间排序
        frames.sort_by_key(|f| f.timestamp);

        info!(
            "加载了 {} 个frames用于会话分析 (session_id={})",
            frames.len(),
            session_id
        );
        Ok(frames)
    }
}

#[async_trait::async_trait]
impl crate::capture::scheduler::SessionProcessor for LLMProcessor {
    async fn process_session(
        &self,
        frames: Vec<crate::capture::ScreenFrame>,
        window: crate::capture::scheduler::SessionWindow,
    ) -> Result<()> {
        // 获取配置
        let config = self.llm_handle.get_config().await?;
        let params = &config.analysis_params;

        // 采样帧
        let sampled_frames = self.sample_frames(&frames, params.frame_sampling_interval as usize);

        // 提取文件路径
        let frame_paths: Vec<String> = sampled_frames.iter().map(|f| f.file_path.clone()).collect();

        // 计算视频时长（分钟）
        let duration = window.end - window.start;
        let duration_minutes = (duration.num_seconds().max(0) as f64 / 60.0).ceil() as u32;

        // 提取所有帧路径用于视频生成
        let all_frame_paths: Vec<String> = frames.iter().map(|f| f.file_path.clone()).collect();

        // 先生成视频（如果配置了视频处理器）
        let mut video_path = None;
        let mut should_persist_frames = true;
        if let Some(ref video_processor) = self.video_processor {
            let app_config = self.settings.get().await;
            if app_config.video_config.auto_generate {
                info!("自动生成会话视频...");

                // 应用帧过滤：每5秒选择一张图片（假设原始截图是1fps）
                let filtered_frame_paths = crate::video::filter_frames_by_interval(
                    all_frame_paths.clone(),
                    5, // 每5秒取一帧
                );

                info!(
                    "视频抽帧：原始 {} 帧，抽样后 {} 帧（每5秒取一帧）",
                    all_frame_paths.len(),
                    filtered_frame_paths.len()
                );

                let file_label = format!(
                    "{}-{}",
                    window.start.format("%Y%m%d%H%M"),
                    window.end.format("%Y%m%d%H%M")
                );
                let output_path = video_processor
                    .output_dir
                    .join(format!("{}.mp4", file_label));

                let mut video_config = crate::video::VideoConfig::default();
                video_config.speed_multiplier = app_config.video_config.speed_multiplier;
                video_config.quality = app_config.video_config.quality;
                video_config.add_timestamp = app_config.video_config.add_timestamp;

                match video_processor
                    .create_summary_video(filtered_frame_paths.clone(), &output_path, &video_config)
                    .await
                {
                    Ok(result) => {
                        info!("视频生成成功: {}", result.file_path);
                        video_path = Some(result.file_path.clone());
// 🔴🔴🔴 修改 1: 注释掉这就话，确保 frames 会被写入数据库，避免成为僵尸文件 🔴🔴🔴
                        // should_persist_frames = false;
// 🔴🔴🔴 修改 2: 注释掉删除文件的循环 🔴🔴🔴
                        /* info!("删除 {} 个原始图片文件...", all_frame_paths.len());
                        for frame_path in &all_frame_paths {
                            if let Err(e) = tokio::fs::remove_file(frame_path).await {
                                error!("删除图片文件失败 {}: {}", frame_path, e);
                            }
                        }
                        info!("原始图片文件已删除");
                        */
                    }
                    Err(e) => {
                        error!("视频生成失败: {}，保留原始图片", e);
                    }
                }
            } else {
                info!("自动生成视频已关闭，跳过视频生成");
            }
        }

        // 检查是否有帧，如果没有帧则不创建会话
        if frame_paths.is_empty() {
            warn!("该时间段没有截图帧，跳过会话创建");
            return Err(anyhow!("没有找到截图帧，无法创建会话"));
        }

        // 先创建会话获取session_id（用于关联LLM调用记录）
        let (device_name, device_type) = crate::storage::get_device_info();
        let temp_session = crate::storage::Session {
            id: None,
            start_time: window.start,
            end_time: window.end,
            title: "处理中...".to_string(),
            summary: "正在分析...".to_string(),
            video_path: video_path.clone(), // 如果已生成视频，这里就有路径了
            tags: "[]".to_string(),
            created_at: None,
            device_name: Some(device_name),
            device_type: Some(device_type),
        };

        let session_id = self.db.insert_session(&temp_session).await?;
        info!("创建临时会话: ID={}", session_id);

        // 记录视频路径，用于错误清理
        let video_path_for_cleanup = video_path.clone();

        // 更新provider的视频路径
        self.llm_handle.set_video_path(video_path.clone()).await?;

        // 设置provider的数据库连接和session_id
        self.llm_handle
            .set_provider_database(self.db.clone(), Some(session_id))
            .await?;

        // 设置视频速率乘数
        let app_config = self.settings.get().await;
        let speed_multiplier = app_config.video_config.speed_multiplier;
        self.llm_handle.set_video_speed(speed_multiplier).await?;

        // 使用两阶段分析：先分段，再生成时间线
        let analysis = {
            match self
                .llm_handle
                .segment_video_and_generate_timeline(frame_paths, duration_minutes, None)
                .await
            {
                Ok(result) => result,
                Err(e) => {
                    // 如果是视频过短错误，清理已创建的资源
                    if e.to_string().contains("VIDEO_TOO_SHORT") {
                        error!("检测到视频过短错误，开始清理资源...");

                        // 1. 删除数据库中的会话记录
                        if let Err(del_err) = self.db.delete_session(session_id).await {
                            error!("删除会话失败 (ID={}): {}", session_id, del_err);
                        } else {
                            info!("已删除会话记录: ID={}", session_id);
                        }

                        // 2. 删除视频文件（如果存在）
                        if let Some(ref vp) = video_path_for_cleanup {
                            if let Err(del_err) = tokio::fs::remove_file(vp).await {
                                error!("删除视频文件失败 {}: {}", vp, del_err);
                            } else {
                                info!("已删除视频文件: {}", vp);
                            }
                        }
                    }

                    return Err(e);
                }
            }
        };

        let TimelineAnalysis {
            mut segments,
            mut timeline_cards,
            segment_call_id,
            timeline_call_id,
        } = analysis;

        for segment in &mut segments {
            let start_abs =
                relative_to_absolute(window.start, window.end, &segment.start_timestamp);
            let end_abs = relative_to_absolute(window.start, window.end, &segment.end_timestamp);
            // 注意：window.start/end 虽然类型是 DateTime<Utc>，但通过 local_now() 存储的是本地时间值
            // 将其重新解释为本地时间并生成带时区标记的 RFC3339 格式（如 2025-10-09T12:54:00+08:00）
            use chrono::{Local, TimeZone};
            segment.start_timestamp = Local
                .from_local_datetime(&start_abs.naive_local())
                .unwrap()
                .to_rfc3339();
            segment.end_timestamp = Local
                .from_local_datetime(&end_abs.naive_local())
                .unwrap()
                .to_rfc3339();
        }

        for card in &mut timeline_cards {
            let start_abs = relative_to_absolute(window.start, window.end, &card.start_time);
            let end_abs = relative_to_absolute(window.start, window.end, &card.end_time);
            // 同样，将其重新解释为本地时间并生成带时区标记的 RFC3339 格式
            use chrono::{Local, TimeZone};
            card.start_time = Local
                .from_local_datetime(&start_abs.naive_local())
                .unwrap()
                .to_rfc3339();
            card.end_time = Local
                .from_local_datetime(&end_abs.naive_local())
                .unwrap()
                .to_rfc3339();

            if let Some(distractions) = card.distractions.as_mut() {
                for distraction in distractions {
                    let d_start =
                        relative_to_absolute(window.start, window.end, &distraction.start_time);
                    let d_end =
                        relative_to_absolute(window.start, window.end, &distraction.end_time);
                    distraction.start_time = Local
                        .from_local_datetime(&d_start.naive_local())
                        .unwrap()
                        .to_rfc3339();
                    distraction.end_time = Local
                        .from_local_datetime(&d_end.naive_local())
                        .unwrap()
                        .to_rfc3339();
                }
            }
        }

        // 保存segments到数据库
        if !segments.is_empty() {
            let segment_records: Vec<crate::storage::VideoSegmentRecord> = segments
                .iter()
                .map(|seg| crate::storage::VideoSegmentRecord {
                    id: None,
                    session_id,
                    llm_call_id: segment_call_id,
                    start_timestamp: seg.start_timestamp.clone(),
                    end_timestamp: seg.end_timestamp.clone(),
                    description: seg.description.clone(),
                    created_at: crate::storage::local_now(),
                })
                .collect();

            self.db.insert_video_segments(&segment_records).await?;
            info!("保存了 {} 个视频分段", segment_records.len());
        }

        // 保存timeline cards到数据库
        if !timeline_cards.is_empty() {
            let card_records: Vec<crate::storage::TimelineCardRecord> = timeline_cards
                .iter()
                .map(|card| {
                    crate::storage::TimelineCardRecord {
                        id: None,
                        session_id,
                        llm_call_id: timeline_call_id,
                        start_time: card.start_time.clone(),
                        end_time: card.end_time.clone(),
                        category: card.category.clone(),
                        subcategory: card.subcategory.clone(),
                        title: card.title.clone(),
                        summary: card.summary.clone(),
                        detailed_summary: card.detailed_summary.clone(),
                        distractions: card
                            .distractions
                            .as_ref()
                            .map(|d| serde_json::to_string(d).unwrap_or_default()),
                        app_sites: serde_json::to_string(&card.app_sites).unwrap_or_default(),
                        video_preview_path: video_path.clone(), // 使用已生成的视频路径
                        created_at: crate::storage::local_now(),
                    }
                })
                .collect();

            self.db.insert_timeline_cards(&card_records).await?;
            info!("保存了 {} 个时间线卡片", card_records.len());
        }

        // 从timeline卡片生成总结（使用第一个卡片的信息）
        let summary = build_session_summary(window.start, window.end, &segments, &timeline_cards);

        // 更新会话信息（之前已经创建了临时会话）
        self.db
            .update_session(
                session_id,
                &summary.title,
                &summary.summary,
                video_path.as_deref(),
                &serde_json::to_string(&summary.tags)?,
            )
            .await?;

        // 保存帧数据（如果没有生成视频则保存路径，否则路径已被删除）
        if should_persist_frames {
            let db_frames: Vec<crate::storage::Frame> = frames
                .iter()
                .map(|f| crate::storage::Frame {
                    id: None,
                    session_id,
                    timestamp: f.timestamp,
                    file_path: f.file_path.clone(),
                })
                .collect();

            self.db.insert_frames(&db_frames).await?;
        }

        info!(
            "会话已保存到数据库: ID={}, 标题={}",
            session_id, summary.title
        );

        // 异步同步到 Notion（不阻塞主流程）
        if let Some(notion_manager) = &self.notion_manager {
            if notion_manager.is_enabled().await {
                // 获取完整的会话信息
                if let Ok(session) = self.db.get_session(session_id).await {
                    info!("触发 Notion 同步：会话 {}", session_id);
                    notion_manager.sync_session_async(session).await;
                }
            }
        }

        // 清理provider的视频路径，避免影响后续会话
        self.llm_handle.set_video_path(None).await?;
        Ok(())
    }
}

impl LLMProcessor {
    /// 采样帧数据
    fn sample_frames(
        &self,
        frames: &[crate::capture::ScreenFrame],
        interval: usize,
    ) -> Vec<crate::capture::ScreenFrame> {
        if frames.is_empty() {
            return vec![];
        }

        let mut sampled = vec![frames[0].clone()]; // 始终包含第一帧

        for i in (interval..frames.len()).step_by(interval) {
            sampled.push(frames[i].clone());
        }

        // 如果最后一帧没有被包含，添加它
        if sampled.last().unwrap().timestamp != frames.last().unwrap().timestamp {
            sampled.push(frames.last().unwrap().clone());
        }

        sampled
    }
}

/// 清理 request_body 中的图片 base64 数据，避免数据库膨胀
///
/// 该函数会遍历 JSON 结构，找到所有图片相关字段并将 base64 数据替换为占位符：
/// - Claude 格式: content[].source.data (type="image") -> "[BASE64_REMOVED]"
/// - Qwen 图片格式: content[].image_url.url (data: 开头) -> "[BASE64_REMOVED]"
/// - Qwen 视频格式: content[].video[] (data: 开头的元素) -> "[BASE64_REMOVED]"
/// - Qwen OSS URL: 保留 http/https 开头的 URL
pub fn sanitize_request_body(value: &Value) -> String {
    fn remove_base64(val: &Value) -> Value {
        match val {
            Value::Object(map) => {
                let mut new_map = serde_json::Map::new();

                for (key, v) in map {
                    match key.as_str() {
                        // Claude 格式：检测 source.data，始终删除 base64
                        "source" => {
                            if let Value::Object(source_map) = v {
                                if source_map.contains_key("data") {
                                    let mut sanitized_source = serde_json::Map::new();
                                    sanitized_source.insert(
                                        "type".to_string(),
                                        source_map
                                            .get("type")
                                            .cloned()
                                            .unwrap_or(Value::String("base64".to_string())),
                                    );
                                    sanitized_source.insert(
                                        "data".to_string(),
                                        Value::String("[BASE64_REMOVED]".to_string()),
                                    );
                                    new_map.insert(key.clone(), Value::Object(sanitized_source));
                                    continue;
                                }
                            }
                            new_map.insert(key.clone(), remove_base64(v));
                        }
                        // Qwen 图片格式：检测 image_url.url
                        "image_url" => {
                            if let Value::Object(img_map) = v {
                                if let Some(url_value) = img_map.get("url") {
                                    if let Some(url_str) = url_value.as_str() {
                                        // 只删除 base64 data URL，保留 http/https URL
                                        if url_str.starts_with("data:") {
                                            let mut sanitized_img = serde_json::Map::new();
                                            sanitized_img.insert(
                                                "url".to_string(),
                                                Value::String("[BASE64_REMOVED]".to_string()),
                                            );
                                            new_map
                                                .insert(key.clone(), Value::Object(sanitized_img));
                                            continue;
                                        }
                                    }
                                }
                            }
                            new_map.insert(key.clone(), remove_base64(v));
                        }
                        // Qwen 视频格式：检测 video 数组
                        "video" => {
                            if let Value::Array(arr) = v {
                                // 检查数组元素是否为 base64 data URL
                                let mut has_base64 = false;
                                let mut cleaned_arr = Vec::new();

                                for item in arr {
                                    if let Some(url_str) = item.as_str() {
                                        if url_str.starts_with("data:") {
                                            // 是 base64，标记并替换
                                            has_base64 = true;
                                            cleaned_arr.push(Value::String(
                                                "[BASE64_REMOVED]".to_string(),
                                            ));
                                        } else {
                                            // 是普通 URL，保留
                                            cleaned_arr.push(item.clone());
                                        }
                                    } else {
                                        cleaned_arr.push(item.clone());
                                    }
                                }

                                if has_base64 {
                                    // 如果包含 base64，使用简化的占位符
                                    let base64_count = cleaned_arr
                                        .iter()
                                        .filter(|v| {
                                            v.as_str()
                                                .map(|s| s == "[BASE64_REMOVED]")
                                                .unwrap_or(false)
                                        })
                                        .count();
                                    new_map.insert(
                                        key.clone(),
                                        Value::String(format!(
                                            "[{} BASE64_IMAGES_REMOVED]",
                                            base64_count
                                        )),
                                    );
                                } else {
                                    // 如果都是普通 URL，保留数组
                                    new_map.insert(key.clone(), Value::Array(cleaned_arr));
                                }
                                continue;
                            }
                            new_map.insert(key.clone(), remove_base64(v));
                        }
                        "images" => {
                            if let Value::Array(arr) = v {
                                // Ollama: images: ["<base64>", ...]
                                let count = arr.len();
                                new_map.insert(
                                    key.clone(),
                                    Value::String(format!("[{} BASE64_IMAGES_REMOVED]", count)),
                                );
                                continue;
                            }
                            new_map.insert(key.clone(), remove_base64(v));
                        }
                        // 其他字段递归处理
                        _ => {
                            new_map.insert(key.clone(), remove_base64(v));
                        }
                    }
                }

                Value::Object(new_map)
            }
            Value::Array(arr) => Value::Array(arr.iter().map(remove_base64).collect()),
            _ => val.clone(),
        }
    }

    serde_json::to_string(&remove_base64(value)).unwrap_or_else(|_| "{}".to_string())
}
