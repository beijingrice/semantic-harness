"use strict";

const el = (id) => document.getElementById(id);
let lastResult = null;

function clear(node) { node.replaceChildren(); }
function element(tag, text, className = "") {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = String(text); // textContent: 不执行模型输出的 HTML
  if (className) node.className = className;
  return node;
}

async function refreshModels() {
  const main = el("model");
  const review = el("verifier-model");
  for (const select of [main, review]) { clear(select); select.append(element("option", "正在加载…")); }
  try {
    const response = await fetch("/api/status");
    const info = await response.json();
    const listed = Array.isArray(info.models) ? info.models : [];
    const configured = [info.configured_model, info.configured_verifier_model].filter(Boolean);
    // A managed API may not implement /models. Show configured IDs even if
    // discovery is skipped; this DOES NOT assert successful API connectivity.
    const options = [...new Set([...listed, ...configured])];
    el("dot").className = "dot " + (info.connected === true ? "up" : "down");
    el("status").textContent = info.connected === true
      ? `模型列表可用 ${info.base_url} · ${listed.length} 个模型`
      : info.connected === null
        ? "未验证 API 连通性（/models 不可用或已关闭）；请使用配置的模型 ID"
        : (info.error || "无法连接模型后端");
    for (const select of [main, review]) {
      clear(select);
      if (!options.length) {
        const opt = element("option", "请先在 .env 配置模型 ID"); opt.value = ""; select.append(opt);
        continue;
      }
      for (const name of options) {
        const opt = element("option", name); opt.value = name; select.append(opt);
      }
    }
    if (options.includes(info.configured_model)) main.value = info.configured_model;
    if (options.includes(info.configured_verifier_model)) review.value = info.configured_verifier_model;
    else if (options.length > 1) {
      review.value = options.find(x => x !== main.value) || options[0];
    }
  } catch (err) {
    el("dot").className = "dot down";
    el("status").textContent = "无法读取后端服务：" + err.message;
    for (const select of [main, review]) clear(select);
  }
}

function chip(keyword, isMissing) {
  const node = element("div", undefined, "chip" + (isMissing ? " missing" : ""));
  node.append(element("strong", keyword.claim));
  const ids = (keyword.paragraph_ids || []).join(", ");
  const label = ids ? `[${keyword.id} · ${ids}]` : `[${keyword.id}]`;
  const evidence = keyword.evidence || "";
  const small = element("small", `${label} ${evidence.length > 180 ? evidence.slice(0, 180) + "…" : evidence}`);
  if (evidence.length > 180) small.title = evidence;
  node.append(small);
  return node;
}

function render(result) {
  const good = result.verdict === "PASS";
  const manual = result.verdict === "REVIEW_REQUIRED";
  el("verdict").className = "verdict " + (good ? "pass" : manual ? "manual" : "fail");
  el("verdict").textContent = good ? "✓ PASS · 关键内容双向覆盖" : manual
    ? "⚠ REVIEW_REQUIRED · 争议尚未解决，需人工复核"
    : "✕ FAIL · 发现缺失或冲突";
  el("summary").textContent = result.summary;
  const missingEn = new Set(result.missing_in_chinese.map((x) => x.id));
  const missingZh = new Set(result.missing_in_english.map((x) => x.id));
  clear(el("en-keys")); clear(el("zh-keys")); clear(el("links"));
  for (const key of result.english_keywords) el("en-keys").append(chip(key, missingEn.has(key.id)));
  for (const key of result.chinese_keywords) el("zh-keys").append(chip(key, missingZh.has(key.id)));
  const enMap = Object.fromEntries(result.english_keywords.map((k) => [k.id, k.claim]));
  const zhMap = Object.fromEntries(result.chinese_keywords.map((k) => [k.id, k.claim]));
  if (!result.links.length) el("links").append(element("p", "未发现可匹配的关键内容。", "muted"));
  for (const link of result.links) {
    const line = element("div", undefined, "relation");
    const enLabel = link.en_ids.map((id) => enMap[id] || id).join(" + ");
    const zhLabel = link.zh_ids.map((id) => zhMap[id] || id).join(" + ");
    line.append(element("strong", enLabel + " ↔ " + zhLabel));
    const status = ({equivalent: "对应", plausible: "合理模糊", related_only: "仅相关、不算覆盖", conflict: "冲突"})[link.relation] || link.relation;
    line.append(element("span", status, "tag " + link.relation));
    line.append(element("p", link.reason, "muted"));
    el("links").append(line);
  }
  clear(el("review-items"));
  const review = result.review;
  if (!review) {
    el("review-items").append(element("p", "仅主模型模式：未执行独立复核。"));
  } else {
    el("review-items").append(element("p", `独立复核意见（${result.verifier_model}）：${review.verdict} — ${review.reason}`));
    if (!review.issues.length) el("review-items").append(element("p", "未提出具体的事实问题。"));
    for (const issue of review.issues) {
      el("review-items").append(element("p", `候选问题 ${issue.kind}: ${issue.reason} [EN: ${issue.en_evidence || "—"}; ZH: ${issue.zh_evidence || "—"}]`));
    }
    for (const warning of result.review_warnings || []) {
      el("review-items").append(element("p", `复核引用警告：${warning}`));
    }
  }
  if (result.arbitration) {
    const arb = result.arbitration;
    el("review-items").append(element("p", `争议仲裁结果：${arb.verdict}（${arb.issue_kind}）— ${arb.reason}`));
    if (arb.en_evidence || arb.zh_evidence)
      el("review-items").append(element("p", `仲裁证据：EN: ${arb.en_evidence || "—"}；ZH: ${arb.zh_evidence || "—"}`));
  }
  if (result.dispute_recheck) {
    el("review-items").append(element("p", "定向异议复核：" + result.dispute_recheck.reason));
    for (const item of result.dispute_recheck.issues) {
      el("review-items").append(element("p", `异议 #${item.issue_index + 1}：${item.status} — ${item.reason}`));
    }
  }
  for (const issue of result.deterministic_issues || []) {
    el("review-items").append(element("p", `确定性校验：${issue}`));
  }
  el("model-used").textContent = `模式：${result.mode} ｜ 主模型：${result.model} ｜ 独立复核：${result.verifier_model || "无"} ｜ 初判：${result.initial_verdict} ｜ 引用：${result.evidence_mode || "quote"} ｜ 仲裁：${result.arbitrated ? "已执行" : "未触发"}`;
  el("diagnostics").textContent = (result.diagnostics || []).join("\n") || "无额外诊断事件";
  el("result").hidden = false;
}

el("refresh").addEventListener("click", refreshModels);
el("run").addEventListener("click", async () => {
  el("error").hidden = true;
  el("result").hidden = true;
  const mode = el("mode").value;
  if (mode === "dual" && el("model").value === el("verifier-model").value) {
    el("error-text").textContent = "请选择两个不同的模型，分别负责初步判断和独立复核。";
    el("error").hidden = false;
    return;
  }
  el("run").disabled = true;
  el("run").textContent = "模型正在分析，请查看推理服务…";
  try {
    const response = await fetch("/api/check", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        english: el("en").value, chinese: el("zh").value,
        model: el("model").value || null,
        verifier_model: el("verifier-model").value || null, mode,
      }),
    });
    const data = await response.json();
    if (!response.ok) {
      const detail = data.detail;
      const message = typeof detail === "string" ? detail : JSON.stringify(detail, null, 2);
      throw new Error(message || `HTTP ${response.status}`);
    }
    lastResult = data;
    render(data);
  } catch (err) {
    el("error-text").textContent = "检查未完成（不是 FAIL）：" + err.message;
    el("error").hidden = false;
  } finally {
    el("run").disabled = false;
    el("run").textContent = "开始检测";
  }
});

el("export").addEventListener("click", () => {
  if (!lastResult) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(lastResult, null, 2)], {type: "application/json"}));
  const anchor = document.createElement("a");
  anchor.href = url; anchor.download = "bilingual-check-result.json"; anchor.click();
  URL.revokeObjectURL(url);
});
refreshModels();
