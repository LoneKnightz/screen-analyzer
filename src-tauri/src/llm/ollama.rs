// src-tauri/src/llm/ollama.rs

use super::plugin::*;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use base64::{engine::general_purpose, Engine as _};
use chrono::Utc;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{debug, info, warn};
use std::sync::Arc;

pub struct OllamaProvider {
    client: Client,
    base_url: String, // e.g. http://localhost:11434
    model: String,    // e.g. qwen3-vl:32b
    configured: bool,
    // 新增：用于记录 LLM 调用、写库等（先放着也行）
    db: Option<Arc<crate::storage::Database>>,
    session_id: Option<i64>,
}

impl OllamaProvider {
    pub fn new(client: Client) -> Self {
        Self {
            client,
            base_url: "http://100.82.18.91:11434".to_string(),
            model: "qwen3-vl:32b".to_string(),
            configured: true, // Ollama 通常不需要 key；有 base_url 就算可用
            db: None,
            session_id: None,
        }
    }

    pub fn set_database(&mut self, db: Arc<crate::storage::Database>) {
        self.db = Some(db);
    }

    pub fn set_session_id(&mut self, session_id: i64) {
        self.session_id = Some(session_id);
    }
    
    fn sample_frames(&self, frames: &[String], max_frames: usize) -> Vec<String> {
        if frames.len() <= max_frames {
            return frames.to_vec();
        }
        let step = (frames.len() / max_frames).max(1);
        frames.iter().step_by(step).take(max_frames).cloned().collect()
    }

    async fn image_to_base64(&self, path: &str) -> Result<String> {
        let bytes = tokio::fs::read(path).await?;
        Ok(general_purpose::STANDARD.encode(bytes))
    }

    fn build_prompt(&self) -> String {
        // 建议沿用你们 Qwen/Claude 的结构化输出要求，确保可解析为 SessionSummary
        r#"
请分析这些屏幕截图，识别用户的活动并输出 严格 JSON（不要多余文本，不要 markdown）。

JSON schema:
{
  "title": "10字以内",
  "summary": "50-100字",
  "tags": [
    {"category":"work|communication|learning|personal|idle|other","confidence":0.0,"keywords":["..."]}
  ],
  "key_moments": [
    {"time":"MM:SS","description":"...","importance":1}
  ],
  "productivity_score": 0,
  "focus_score": 0
}

只返回 JSON。
"#.trim().to_string()
    }

    async fn call_ollama_chat(&self, images_b64: Vec<String>) -> Result<String> {
        let url = format!("{}/api/chat", self.base_url.trim_end_matches('/'));

        #[derive(Serialize)]
        struct Msg {
            role: String,
            content: String,
            #[serde(skip_serializing_if = "Option::is_none")]
            images: Option<Vec<String>>,
        }
        #[derive(Serialize)]
        struct Req {
            model: String,
            stream: bool,
            messages: Vec<Msg>,
        }
        #[derive(Deserialize)]
        struct Resp {
            message: RespMsg,
        }
        #[derive(Deserialize)]
        struct RespMsg {
            content: String,
        }

        let req = Req {
            model: self.model.clone(),
            stream: false,
            messages: vec![Msg {
                role: "user".to_string(),
                content: self.build_prompt(),
                images: Some(images_b64),
            }],
        };

        let resp: Resp = self
            .client
            .post(url)
            .json(&req)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;

        Ok(resp.message.content)
    }

    fn extract_json_text<'a>(&self, raw: &'a str) -> &'a str {
        // 兼容模型偶尔返回 ```json ... ``` 的情况
        let s = raw.trim();
        if let Some(pos) = s.find("```") {
            // 简单剥离 code fence（够用）
            let s2 = s[pos..].trim_start_matches("```json").trim_start_matches("```");
            if let Some(end) = s2.find("```") {
                return s2[..end].trim();
            }
        }
        s
    }

    fn parse_session_summary(&self, raw: &str) -> Result<SessionSummary> {
        let json_text = self.extract_json_text(raw);
        // 将 JSON 解析为 Value，以便我们可以在转换 Struct 之前修改它
        let mut v: Value = serde_json::from_str(json_text)
            .map_err(|e| anyhow!("Ollama 返回不是合法 JSON: {e}; raw={}", raw))?;

        // ✅ 关键修复：手动注入缺失的时间字段
        // LLM 不知道绝对时间，所以我们在这里给一个默认值（当前时间）
        // 后续业务逻辑通常会用真实的会话时间覆盖它
        if let Some(obj) = v.as_object_mut() {
            let now = Utc::now();
            if !obj.contains_key("start_time") {
                obj.insert("start_time".to_string(), serde_json::to_value(now)?);
            }
            if !obj.contains_key("end_time") {
                obj.insert("end_time".to_string(), serde_json::to_value(now)?);
            }
        }

        // 现在再转换为 SessionSummary Struct
        let mut summary: SessionSummary = serde_json::from_value(v)?;
        
        let now = Utc::now();
        if summary.start_time > summary.end_time {
            summary.start_time = now;
            summary.end_time = now;
        }
        Ok(summary)
    }
}

#[async_trait]
impl LLMProvider for OllamaProvider {
    fn as_any(&mut self) -> &mut dyn std::any::Any {
        self
    }

    async fn analyze_frames(&self, frames: Vec<String>) -> Result<SessionSummary> {
        if !self.configured {
            return Err(anyhow!("Ollama provider 未配置"));
        }

        info!("Ollama: 开始分析 {} 帧", frames.len());

        // 采样：沿用你们 analysis_params 的默认策略（最多 30）
        let sampled = self.sample_frames(&frames, 30);
        debug!("Ollama: 采样后 {} 帧", sampled.len());

        // 编码
        let mut images_b64 = Vec::new();
        for path in sampled {
            match self.image_to_base64(&path).await {
                Ok(b64) => images_b64.push(b64),
                Err(e) => warn!("Ollama: 编码失败 path={} err={}", path, e),
            }
        }
        if images_b64.is_empty() {
            return Err(anyhow!("没有可用的图片帧用于分析"));
        }

        // 调用
        let raw = self.call_ollama_chat(images_b64).await?;
        self.parse_session_summary(&raw)
    }

    fn name(&self) -> &str {
        "ollama"
    }

    fn configure(&mut self, config: serde_json::Value) -> Result<()> {
        if let Some(base_url) = config.get("base_url").and_then(|v| v.as_str()) {
            self.base_url = base_url.to_string();
        }
        if let Some(model) = config.get("model").and_then(|v| v.as_str()) {
            self.model = model.to_string();
        }
        // base_url 至少要有
        self.configured = !self.base_url.trim().is_empty();
        Ok(())
    }

    fn is_configured(&self) -> bool {
        self.configured
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            vision_support: true,
            batch_analysis: true,
            streaming: false,
            max_input_tokens: 128000,
            supported_image_formats: vec!["jpg".to_string(), "jpeg".to_string(), "png".to_string()],
        }
    }
}