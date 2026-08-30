const API = {
  chatStream: "/api/chat/stream",
  chatFusionStream: "/api/chat/fusion/stream",
  taskFusion: "/api/tasks/fusion",
  taskBatchFusion: "/api/tasks/fusion/batch",
  taskStatus: (id) => `/api/tasks/${id}`,
  batchFusionStream: "/api/fusion/batch/stream",
  reportGenerate: "/api/report/generate",
  fusionCurrent: "/api/fusion/current",
  sessionClear: "/api/session/clear",
  systemInfo: "/api/system/info",
  sessions: "/api/sessions",
  sessionNew: "/api/session/new",
  session: (id) => `/api/session/${id}`,
  sessionMessages: (id) => `/api/session/${id}/messages`,
};

const $ = (id) => document.getElementById(id);

const chatArea = $("chatArea");
const workflowList = $("workflowList");
const promptInput = $("promptInput");
const opticalFile = $("opticalFile");
const sarFile = $("sarFile");
const opticalName = $("opticalName");
const sarName = $("sarName");
const useTile = $("useTile");
const tileSize = $("tileSize");
const overlap = $("overlap");
const generateReport = $("generateReport");
const systemStatus = $("systemStatus");

let sessionId = localStorage.getItem("optisar_session_id") || crypto.randomUUID().replaceAll("-", "");
localStorage.setItem("optisar_session_id", sessionId);
let localSessions = JSON.parse(localStorage.getItem("optisar_recent_sessions") || "[]");
let localSessionMessages = JSON.parse(localStorage.getItem("optisar_session_messages") || "{}");

function escapeHtml(text = "") {
  return String(text).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}
function escapeAttr(text = "") {
  return escapeHtml(text).replaceAll('"', "&quot;");
}
function nowText() { return new Date().toLocaleString(); }
function scrollDown() { chatArea.scrollTop = chatArea.scrollHeight; }
function removeWelcome() { const w = chatArea.querySelector(".welcome-card"); if (w) w.remove(); }
function welcomeHtml() {
  return `
    <div class="welcome-card">
      <div class="welcome-kicker">SELECT · RETRIEVE · FUSE</div>
      <h3>从需求到成像方案，一处完成</h3>
      <p>可以直接询问设备选型，也可以上传同一场景的红外与可见光图像执行融合。</p>
      <div class="starter-grid">
        <button class="starter-prompt" type="button" data-prompt="推荐支持SDK、分辨率不低于640×512的红外设备"><span>设备选型</span><strong>按指标推荐红外设备</strong></button>
        <button class="starter-prompt" type="button" data-prompt="对比知识库中的红外、可见光和双模态设备特点与适用场景"><span>产品对比</span><strong>比较模态与应用场景</strong></button>
        <button class="starter-prompt" type="button" data-prompt="解释图像融合中的Qabf、MI和SSIM指标"><span>指标问答</span><strong>理解融合质量指标</strong></button>
      </div>
    </div>`;
}

function saveLocalSessions() {
  localStorage.setItem("optisar_recent_sessions", JSON.stringify(localSessions.slice(0, 20)));
}

function saveLocalMessages() {
  localStorage.setItem("optisar_session_messages", JSON.stringify(localSessionMessages));
}

function currentLocalMessages() {
  if (!localSessionMessages[sessionId]) localSessionMessages[sessionId] = [];
  return localSessionMessages[sessionId];
}

function rememberMessage(role, content) {
  if (!content) return;
  const messages = currentLocalMessages();
  messages.push({ role, content, time: new Date().toISOString() });
  localSessionMessages[sessionId] = messages.slice(-80);
  saveLocalMessages();
}

function persistMessageToServer(id, role, content) {
  if (!id || !content) return;
  fetch(API.sessionMessages(id), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ role, content }),
  }).then(refreshSessionList).catch(() => {});
}

function rememberAssistantFinal(content) {
  if (!content) return;
  const messages = currentLocalMessages();
  const last = messages[messages.length - 1];
  if (last && last.role === "assistant" && last.pending) {
    last.content = content;
    delete last.pending;
  } else {
    messages.push({ role: "assistant", content, time: new Date().toISOString() });
  }
  localSessionMessages[sessionId] = messages.slice(-80);
  saveLocalMessages();
}

function sessionTitleFromText(text = "") {
  const clean = String(text).replace(/\s+/g, " ").trim();
  return clean ? (clean.length > 28 ? `${clean.slice(0, 28)}...` : clean) : "新会话";
}

function upsertLocalSession(id, title = "新会话", updatedAt = new Date().toISOString()) {
  localSessions = localSessions.filter((item) => item.session_id !== id);
  localSessions.unshift({ session_id: id, title, updated_at: updatedAt });
  saveLocalSessions();
  renderSessionList();
}

function mergeSessionRows(serverRows = []) {
  const map = new Map();
  localSessions.forEach((item) => map.set(item.session_id, item));
  serverRows.forEach((item) => map.set(item.session_id, { ...map.get(item.session_id), ...item }));
  Object.entries(localSessionMessages).forEach(([id, messages]) => {
    if (!Array.isArray(messages) || !messages.length || map.has(id)) return;
    const firstUser = messages.find((m) => m.role === "user");
    map.set(id, {
      session_id: id,
      title: sessionTitleFromText(firstUser?.content || "新会话"),
      updated_at: messages[messages.length - 1]?.time || new Date().toISOString(),
      message_count: messages.length,
    });
  });
  localSessions = Array.from(map.values())
    .filter((item) => {
      const messages = localSessionMessages[item.session_id] || [];
      return item.session_id === sessionId || item.message_count > 0 || messages.length > 0;
    })
    .sort((a, b) => String(b.updated_at || "").localeCompare(String(a.updated_at || "")))
    .slice(0, 20);
  saveLocalSessions();
}

function renderSessionList() {
  const box = $("sessionList");
  if (!box) return;
  const keyword = ($("sessionSearch")?.value || "").trim().toLowerCase();
  const rows = localSessions.filter((item) => !keyword || String(item.title || "").toLowerCase().includes(keyword));
  if (!rows.length) {
    box.innerHTML = `<div class="session-item active"><span class="session-title">暂无历史会话</span><span class="session-meta">发送一条消息后会出现在这里</span></div>`;
    return;
  }
  box.innerHTML = rows.map((item) => `
    <button class="session-item ${item.session_id === sessionId ? "active" : ""}" type="button" data-session-id="${escapeHtml(item.session_id)}">
      <span class="session-title">${escapeHtml(item.title || "新会话")}</span>
      <span class="session-meta">${escapeHtml(item.updated_at || "")}</span>
    </button>
  `).join("");
  box.querySelectorAll("[data-session-id]").forEach((btn) => {
    btn.addEventListener("click", () => switchSession(btn.dataset.sessionId));
  });
}

async function refreshSessionList() {
  try {
    const res = await fetch(API.sessions);
    const data = await res.json();
    mergeSessionRows(data.sessions || []);
  } catch (_) {
    upsertLocalSession(sessionId, localSessions.find((s) => s.session_id === sessionId)?.title || "当前会话");
  }
  renderSessionList();
}

async function bootstrapSessionUI() {
  await refreshSessionList();
  const localMessages = localSessionMessages[sessionId] || [];
  const known = localSessions.some((item) => item.session_id === sessionId);
  if (localMessages.length || known) {
    await restoreSession(sessionId);
  } else {
    renderSessionList();
  }
}

function setActiveNav(activeId) {
  document.querySelectorAll(".nav-item").forEach((el) => el.classList.remove("active"));
  const active = $(activeId);
  if (active) active.classList.add("active");
}
function setHeader(title, desc) {
  $("pageTitle").textContent = title;
  $("pageDesc").textContent = desc;
}

function clearWorkflow() {
  workflowList.innerHTML = `<div class="flow-empty"><strong>等待任务</strong><span>发送问题后，这里会显示检索、重排和工具调用过程。</span></div>`;
}

function setWorkflowExpanded(expanded) {
  const shell = $("appShell");
  const toggle = $("toggleWorkflowBtn");
  if (!shell || !toggle) return;
  shell.classList.toggle("workflow-collapsed", !expanded);
  toggle.textContent = expanded ? "收起流程" : "展开流程";
  toggle.setAttribute("aria-expanded", String(expanded));
  localStorage.setItem("mambadfuse_workflow_expanded", String(expanded));
}
function addFlow(stage, detail = "", tool = "") {
  const empty = workflowList.querySelector(".flow-empty");
  if (empty) empty.remove();
  const item = document.createElement("div");
  item.className = "flow-item";
  item.innerHTML = `
    <div class="flow-stage">${escapeHtml(stage)}</div>
    <div class="flow-detail">${escapeHtml(detail)}</div>
    ${tool ? `<div class="flow-tool">工具：${escapeHtml(tool)}</div>` : ""}
  `;
  workflowList.appendChild(item);
  workflowList.scrollTop = workflowList.scrollHeight;
}
function addTrace(trace = []) {
  if (!trace || !trace.length) {
    addFlow("工具轨迹", "本次没有返回具体工具调用轨迹，可能是直接回答。", "");
    return;
  }
  trace.forEach((step, i) => {
    addFlow(`工具调用 ${i + 1}: ${step.tool || "unknown_tool"}`, `输入：${JSON.stringify(step.input ?? "").slice(0, 160)}\n输出：${JSON.stringify(step.output ?? "").slice(0, 240)}`, step.tool || "");
  });
}

function addMessage(role, content = "", extraNode = null) {
  removeWelcome();
  const hasResult = extraNode?.classList?.contains("result-wrap");
  const row = document.createElement("div");
  row.className = `message ${role}`;
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "你" : "AI";
  const box = document.createElement("div");
  box.className = "message-box";
  if (hasResult) box.classList.add("result-message");
  const meta = document.createElement("div");
  meta.className = "message-meta";
  meta.textContent = `${role === "user" ? "用户" : "MambaDFuse-Agent"} · ${nowText()}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (hasResult) bubble.classList.add("result-bubble");
  bubble.innerHTML = escapeHtml(content).replaceAll("\n", "<br>");
  if (extraNode) bubble.appendChild(extraNode);
  box.appendChild(meta);
  box.appendChild(bubble);
  row.appendChild(avatar);
  row.appendChild(box);
  chatArea.appendChild(row);
  scrollDown();
  return bubble;
}

function appendToBubble(bubble, text) {
  bubble.innerHTML += escapeHtml(text).replaceAll("\n", "<br>");
  scrollDown();
}

function metricValue(v) {
  if (typeof v === "number") return Number(v).toFixed(4);
  const n = Number(v);
  if (!Number.isNaN(n)) return n.toFixed(4);
  return String(v ?? "");
}

function downloadNameFromUrl(url = "", title = "result") {
  const cleanUrl = String(url).split("?")[0];
  const rawName = decodeURIComponent(cleanUrl.split("/").pop() || `${title}.png`);
  const safeTitle = String(title).replace(/[\\/:*?"<>|]+/g, "_");
  return rawName.includes(".") ? rawName.replace(/[\\/:*?"<>|]+/g, "_") : `${safeTitle}.png`;
}

function createFusionBlock(result, reportText = "") {
  const wrap = document.createElement("div");
  wrap.className = "result-wrap";
  if (result) {
    const urls = result.urls || {};
    const imageItems = [
      ["可见光图像", urls.optical_url || result.optical_url],
      ["红外图像", urls.sar_url || result.sar_url],
      ["融合图像", urls.fused_color_url || result.fused_color_url],
      ["可见光边缘", urls.edge_optical_url || result.edge_optical_url],
      ["红外边缘", urls.edge_sar_url || result.edge_sar_url],
      ["融合边缘", urls.edge_fused_url || result.edge_fused_url],
      ["差异图", urls.diff_url || result.diff_url],
    ].filter(([, url]) => !!url);

    if (imageItems.length) {
      const grid = document.createElement("div");
      grid.className = "image-grid";
      imageItems.forEach(([title, url]) => {
        const card = document.createElement("div");
        card.className = "image-card";
        const fileName = downloadNameFromUrl(url, title);
        card.innerHTML = `
          <div class="image-title">${escapeHtml(title)}<span>点击图片下载</span></div>
          <a class="image-download" href="${escapeAttr(url)}" download="${escapeAttr(fileName)}" title="点击下载${escapeAttr(title)}">
            <img src="${escapeAttr(url)}" alt="${escapeAttr(title)}" />
          </a>
        `;
        grid.appendChild(card);
      });
      wrap.appendChild(grid);
    }

    const metrics = result.metrics || {};
    if (Object.keys(metrics).length) {
      const panel = document.createElement("div");
      panel.className = "metrics-panel";
      panel.innerHTML = `<h4>融合质量指标</h4>`;
      const grid = document.createElement("div");
      grid.className = "metrics-grid";
      Object.entries(metrics).forEach(([k, v]) => {
        const item = document.createElement("div");
        item.className = "metric-item";
        item.innerHTML = `<span>${escapeHtml(k)}</span><strong>${escapeHtml(metricValue(v))}</strong>`;
        grid.appendChild(item);
      });
      panel.appendChild(grid);
      wrap.appendChild(panel);
    }
  }
  if (reportText) {
    const report = document.createElement("div");
    report.className = "report-panel";
    report.innerHTML = `<h4>分析报告</h4><pre>${escapeHtml(reportText)}</pre>`;
    wrap.appendChild(report);
  }
  return wrap;
}

function createBatchFusionBlock(results = [], reportText = "") {
  const wrap = document.createElement("div");
  wrap.className = "result-wrap";
  if (results.length) {
    const rows = results.map((r) => {
      const metrics = r.metrics || {};
      return `
        <tr>
          <td>${escapeHtml(r._pair_index ?? "")}</td>
          <td>${escapeHtml(r._optical_name || "")}</td>
          <td>${escapeHtml(r._sar_name || "")}</td>
          <td>${escapeHtml(metricValue(metrics["综合质量评分"] ?? 0))}</td>
          <td>${escapeHtml(metricValue(metrics["MI 互信息"] ?? 0))}</td>
          <td>${escapeHtml(metricValue(metrics["SSIM 结构相似性"] ?? 0))}</td>
          <td>${escapeHtml(metricValue(metrics["Qabf 边缘保持"] ?? 0))}</td>
        </tr>
      `;
    }).join("");
    const table = document.createElement("div");
    table.className = "batch-table-wrap";
    table.innerHTML = `
      <h4>批量融合指标汇总</h4>
      <table class="batch-table">
        <thead><tr><th>序号</th><th>可见光图像</th><th>红外图像</th><th>评分</th><th>MI</th><th>SSIM</th><th>Qabf</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    `;
    wrap.appendChild(table);

    results.forEach((result) => {
      const details = document.createElement("details");
      details.className = "batch-detail";
      const score = result.metrics?.["综合质量评分"] ?? 0;
      details.innerHTML = `<summary>第 ${escapeHtml(result._pair_index)} 组：${escapeHtml(result._optical_name || "")} + ${escapeHtml(result._sar_name || "")}，评分 ${escapeHtml(metricValue(score))}/100</summary>`;
      details.appendChild(createFusionBlock(result, ""));
      wrap.appendChild(details);
    });
  }
  if (reportText) {
    const report = document.createElement("div");
    report.className = "report-panel";
    report.innerHTML = `<h4>批量分析报告</h4><pre>${escapeHtml(reportText)}</pre>`;
    wrap.appendChild(report);
  }
  return wrap;
}

function setBusy(isBusy) {
  const button = $("sendSmartBtn");
  button.disabled = isBusy;
  button.innerHTML = isBusy
    ? "<span>处理中...</span><small>请稍候</small>"
    : "<span>发送</span><small>Ctrl + Enter</small>";
}
function setComposerVisible(visible) {
  $("composer").style.display = visible ? "" : "none";
}
function updateFileNames() {
  const optical = [...(opticalFile.files || [])];
  const sar = [...(sarFile.files || [])];
  opticalName.textContent = optical.length ? `${optical.length} 张：${optical.map((f) => f.name).slice(0, 3).join("，")}${optical.length > 3 ? "..." : ""}` : "未选择";
  sarName.textContent = sar.length ? `${sar.length} 张：${sar.map((f) => f.name).slice(0, 3).join("，")}${sar.length > 3 ? "..." : ""}` : "未选择";
  renderBatchPreview();
}
opticalFile.addEventListener("change", updateFileNames);
sarFile.addEventListener("change", updateFileNames);

function selectedBatchFiles() {
  const optical = [...(opticalFile?.files || [])];
  const sar = [...(sarFile?.files || [])];
  const pairMode = $("batchPairMode")?.value || "name";
  const sortedOptical = pairMode === "name" ? [...optical].sort((a, b) => a.name.localeCompare(b.name, "zh-Hans-CN")) : optical;
  const sortedSar = pairMode === "name" ? [...sar].sort((a, b) => a.name.localeCompare(b.name, "zh-Hans-CN")) : sar;
  return { optical, sar, sortedOptical, sortedSar, pairMode };
}

function renderBatchPreview() {
  const box = $("batchPairPreview");
  if (!box) return;
  const { optical, sar, sortedOptical, sortedSar } = selectedBatchFiles();
  const pairCount = Math.min(sortedOptical.length, sortedSar.length);
  if (!optical.length && !sar.length) {
    box.innerHTML = "";
    $("batchStatus").textContent = "等待上传图像。";
    $("batchProgressBar").style.width = "0%";
    return;
  }
  if (optical.length <= 1 && sar.length <= 1) {
    box.innerHTML = "";
    $("batchStatus").textContent = optical.length && sar.length ? "已选择一对图像，将执行单次融合。" : "请同时上传可见光图像和红外图像。";
    $("batchProgressBar").style.width = "0%";
    return;
  }
  const rows = Array.from({ length: pairCount }, (_, i) => `
    <tr><td>${i + 1}</td><td>${escapeHtml(sortedOptical[i].name)}</td><td>${escapeHtml(sortedSar[i].name)}</td></tr>
  `).join("");
  box.innerHTML = `
    <div class="batch-table-wrap">
      <h4>配对预览</h4>
      <table class="batch-table">
        <thead><tr><th>序号</th><th>可见光图像</th><th>红外图像</th></tr></thead>
        <tbody>${rows || `<tr><td colspan="3">暂无可配对图像</td></tr>`}</tbody>
      </table>
    </div>
  `;
  const extra = optical.length !== sar.length ? `数量不一致，将只处理前 ${pairCount} 组。` : `可处理 ${pairCount} 组。`;
  $("batchStatus").textContent = `可见光 ${optical.length} 张，红外 ${sar.length} 张，${extra}`;
}

async function fetchSystemInfo() {
  try {
    const res = await fetch(API.systemInfo);
    const data = await res.json();
    systemStatus.innerHTML = `后端在线<br>时间：${escapeHtml(data.time || "")}<br>输出目录：${escapeHtml(data.output_dir || "")}`;
  } catch (_) {
    systemStatus.textContent = "后端状态读取失败";
  }
}

async function readNdjsonStream(res, onEvent) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        onEvent(JSON.parse(line));
      } catch (e) {
        console.warn("bad ndjson", line, e);
      }
    }
  }
}

async function waitForTask(statusUrl, onProgress, timeoutMs = 30 * 60 * 1000) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    const res = await fetch(statusUrl, { cache: "no-store" });
    if (!res.ok) throw new Error((await res.text()) || "任务状态读取失败");
    const task = await res.json();
    onProgress(task);
    if (task.status === "succeeded") return task.result || {};
    if (["failed", "cancelled"].includes(task.status)) {
      throw new Error(task.error || `任务已${task.status === "cancelled" ? "取消" : "失败"}`);
    }
    await new Promise((resolve) => setTimeout(resolve, 900));
  }
  throw new Error("任务等待超时，请稍后通过任务编号查询状态");
}

async function sendNormalChat(prompt) {
  clearWorkflow();
  let assistantBubble = addMessage("assistant", "");
  let assistantText = "";
  let finalData = null;

  const res = await fetch(API.chatStream, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      question: prompt,
      prompt,
      use_tile: useTile.checked,
      tile: Number(tileSize.value || 256),
      overlap: Number(overlap.value || 32),
    }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || "普通问答失败");
  }

  await readNdjsonStream(res, (ev) => {
    if (ev.type === "workflow") addFlow(ev.stage || "流程", ev.detail || "", ev.tool || "");
    if (ev.type === "trace") addTrace(ev.trace || []);
    if (ev.type === "answer_chunk") {
      assistantText += ev.text || "";
      appendToBubble(assistantBubble, ev.text || "");
    }
    if (ev.type === "final") finalData = ev;
    if (ev.type === "error") appendToBubble(assistantBubble, `\n执行失败：${ev.message}`);
  });

  if (finalData && (finalData.fusion_result || finalData.report_text)) {
    assistantBubble.appendChild(createFusionBlock(finalData.fusion_result, finalData.report_text || ""));
  }
  rememberAssistantFinal(finalData?.answer || assistantText);
}

async function sendFusionChat(prompt, optical, sar) {
  clearWorkflow();
  const assistantBubble = addMessage("assistant", "正在提交融合任务...");

  const fd = new FormData();
  fd.append("session_id", sessionId);
  fd.append("prompt", prompt);
  fd.append("generate_report", generateReport.checked ? "true" : "false");
  fd.append("use_tile", useTile.checked ? "true" : "false");
  fd.append("tile", String(Number(tileSize.value || 256)));
  fd.append("overlap", String(Number(overlap.value || 32)));
  fd.append("optical_file", optical);
  fd.append("infrared_file", sar);

  const res = await fetch(API.taskFusion, { method: "POST", body: fd });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || "融合任务提交失败");
  }
  const accepted = await res.json();
  assistantBubble.textContent = `任务已进入 Redis 队列（${accepted.task_id}）`;
  addFlow("任务已入队", `任务编号：${accepted.task_id}`, API.taskFusion);

  const result = await waitForTask(accepted.status_url || API.taskStatus(accepted.task_id), (task) => {
    const percent = Math.round(Number(task.progress || 0) * 100);
    const stage = task.stage || (task.status === "queued" ? "等待 GPU Worker" : "正在处理");
    assistantBubble.textContent = `${stage}${percent ? `（${percent}%）` : ""}`;
  });
  assistantBubble.textContent = "融合完成。";
  assistantBubble.appendChild(createFusionBlock(result.fusion_result, result.report_text || ""));
  addFlow("任务完成", "GPU Worker 已返回融合结果", API.taskStatus(accepted.task_id));
  rememberAssistantFinal("融合完成。结果已由 Redis 任务队列返回。");

  opticalFile.value = "";
  sarFile.value = "";
  updateFileNames();
}

async function sendBatchFusionChat(prompt, sortedOptical, sortedSar, pairMode) {
  clearWorkflow();
  const pairCount = Math.min(sortedOptical.length, sortedSar.length);
  let assistantBubble = addMessage("assistant", `开始批量融合，共 ${pairCount} 组图像。`);
  let finalData = null;
  const bar = $("batchProgressBar");

  const fd = new FormData();
  fd.append("session_id", sessionId);
  fd.append("generate_report", generateReport.checked ? "true" : "false");
  fd.append("use_tile", useTile.checked ? "true" : "false");
  fd.append("tile", String(Number(tileSize.value || 256)));
  fd.append("overlap", String(Number(overlap.value || 32)));
  fd.append("pair_mode", pairMode);
  sortedOptical.forEach((file) => fd.append("optical_files", file));
  sortedSar.forEach((file) => fd.append("sar_files", file));

  addFlow("提交批量任务", `准备上传 ${pairCount} 组图像`, API.taskBatchFusion);
  $("batchStatus").textContent = "正在上传并进入队列...";
  bar.style.width = "0%";

  const res = await fetch(API.taskBatchFusion, { method: "POST", body: fd });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || "批量融合任务提交失败");
  }
  const accepted = await res.json();
  assistantBubble.textContent = `批量任务已进入 Redis 队列（${accepted.task_id}）`;
  addFlow("任务已入队", `任务编号：${accepted.task_id}`, API.taskBatchFusion);

  finalData = await waitForTask(accepted.status_url || API.taskStatus(accepted.task_id), (task) => {
    const percent = Math.round(Number(task.progress || 0) * 100);
    bar.style.width = `${percent}%`;
    $("batchStatus").textContent = task.stage || (task.status === "queued" ? "等待 GPU Worker" : "正在批量融合");
    assistantBubble.textContent = `${$("batchStatus").textContent}${percent ? `（${percent}%）` : ""}`;
  });

  if (finalData) {
    assistantBubble.textContent = "";
    bar.style.width = "100%";
    $("batchStatus").textContent = finalData.answer || "批量融合完成。";
    assistantBubble.appendChild(createBatchFusionBlock(finalData.batch_results || [], finalData.report_text || ""));
    rememberAssistantFinal(finalData.answer || `批量融合完成：${pairCount} 组图像。`);
  }

  opticalFile.value = "";
  sarFile.value = "";
  updateFileNames();
}

async function smartSend() {
  const prompt = promptInput.value.trim();
  const { optical, sar, sortedOptical, sortedSar, pairMode } = selectedBatchFiles();
  const opticalOne = sortedOptical[0] || null;
  const sarOne = sortedSar[0] || null;
  const pairCount = Math.min(sortedOptical.length, sortedSar.length);
  const isBatch = optical.length > 1 || sar.length > 1;

  if (!prompt && !optical.length && !sar.length) {
    addMessage("assistant", "请输入问题，或者同时上传可见光图像和红外图像。");
    return;
  }
  if ((optical.length && !sar.length) || (!optical.length && sar.length)) {
    addMessage("assistant", "如果要进行图像融合，请同时上传可见光图像和红外图像；如果只是普通问答，请清空图片选择后直接发送。");
    return;
  }
  if (isBatch && pairCount <= 0) {
    addMessage("assistant", "批量融合需要至少一张可见光图像和一张红外图像。");
    return;
  }

  const finalPrompt = prompt || (isBatch ? "请批量融合我上传的可见光图像组和红外图像组，并给出简要分析。" : "请融合我上传的可见光图像和红外图像，并给出简要分析。");
  const userText = optical.length && sar.length
    ? `${finalPrompt}\n[已附加图片：可见光 ${optical.length} 张 + 红外 ${sar.length} 张；配对方式：${pairMode === "name" ? "按文件名" : "按上传顺序"}]`
    : finalPrompt;
  addMessage("user", userText);
  rememberMessage("user", userText);
  upsertLocalSession(sessionId, sessionTitleFromText(finalPrompt), new Date().toISOString());
  promptInput.value = "";
  setBusy(true);

  try {
    if (isBatch) await sendBatchFusionChat(finalPrompt, sortedOptical, sortedSar, pairMode);
    else if (opticalOne && sarOne) await sendFusionChat(finalPrompt, opticalOne, sarOne);
    else await sendNormalChat(finalPrompt);
  } catch (err) {
    addMessage("assistant", `执行失败：${err.message}`);
    addFlow("执行失败", err.message, "");
  } finally {
    setBusy(false);
    refreshSessionList();
  }
}

async function generateCurrentReport() {
  addMessage("user", "生成当前报告");
  setBusy(true);
  try {
    const res = await fetch(API.reportGenerate, { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || data.message || "生成报告失败");
    addMessage("assistant", "已生成当前报告。", createFusionBlock(null, data.report || ""));
    addFlow("生成报告", "调用 /api/report/generate", "build_single_report_markdown");
  } catch (err) {
    addMessage("assistant", `生成报告失败：${err.message}`);
  } finally {
    setBusy(false);
  }
}

async function loadCurrentFusion() {
  try {
    const res = await fetch(API.fusionCurrent);
    const data = await res.json();
    const batchResults = data.batch_results || [];
    const result = data.last_result || null;
    if (batchResults.length) {
      addMessage("assistant", `已读取最近批量融合结果：${batchResults.length} 组。`, createBatchFusionBlock(batchResults, ""));
      return;
    }
    if (!result) {
      addMessage("assistant", "当前还没有最近融合结果。");
      return;
    }
    addMessage("assistant", "已读取最近一次融合结果。", createFusionBlock(result, ""));
  } catch (err) {
    addMessage("assistant", `读取失败：${err.message}`);
  }
}

async function restoreSession(targetSessionId = sessionId) {
  try {
    const res = await fetch(API.session(targetSessionId));
    const data = res.ok ? await res.json() : {};
    sessionId = targetSessionId;
    localStorage.setItem("optisar_session_id", sessionId);
    const serverMessages = Array.isArray(data.messages) ? data.messages : [];
    const localMessages = localSessionMessages[sessionId] || [];
    const messages = serverMessages.length >= localMessages.length ? serverMessages : localMessages;
    if (serverMessages.length) {
      localSessionMessages[sessionId] = serverMessages.map((m) => ({
        role: m.role || "assistant",
        content: m.content || "",
        time: m.time || data.updated_at || new Date().toISOString(),
      }));
      saveLocalMessages();
    }
    const firstUser = messages.find((m) => m.role === "user");
    upsertLocalSession(sessionId, firstUser?.content ? sessionTitleFromText(firstUser.content) : "新会话", data.updated_at || messages[messages.length - 1]?.time || new Date().toISOString());
    focusChat();
    if (!messages.length) {
      chatArea.innerHTML = welcomeHtml();
      return;
    }
    chatArea.innerHTML = "";
    messages.forEach((m) => addMessage(m.role || "assistant", m.content || ""));
    if (data.last_result || data.last_report) {
      addMessage("assistant", "已恢复最近一次融合结果。", createFusionBlock(data.last_result, data.last_report || ""));
    }
  } catch (err) {
    const messages = localSessionMessages[targetSessionId] || [];
    if (messages.length) {
      sessionId = targetSessionId;
      localStorage.setItem("optisar_session_id", sessionId);
      focusChat();
      chatArea.innerHTML = "";
      messages.forEach((m) => addMessage(m.role || "assistant", m.content || ""));
      renderSessionList();
      return;
    }
    addMessage("assistant", `恢复会话失败：${err.message}`);
  }
}

async function newSession() {
  await refreshSessionList();
  let nextId = crypto.randomUUID().replaceAll("-", "");
  try {
    const res = await fetch(API.sessionNew, { method: "POST" });
    const data = await res.json();
    if (res.ok && data.session_id) nextId = data.session_id;
  } catch (_) {}
  sessionId = nextId;
  localStorage.setItem("optisar_session_id", sessionId);
  renderSessionList();
  clearWorkflow();
  focusChat();
  chatArea.innerHTML = welcomeHtml();
}

async function clearSession() {
  try { await fetch(API.sessionClear, { method: "POST" }); } catch (_) {}
  localSessions = [];
  localSessionMessages = {};
  saveLocalSessions();
  saveLocalMessages();
  await newSession();
}

async function switchSession(id) {
  if (!id || id === sessionId) {
    renderSessionList();
    return;
  }
  await restoreSession(id);
}

function focusChat() {
  setComposerVisible(true);
  setActiveNav("navChat");
  setHeader("智能会话", "检索设备资料、对比产品参数，或上传红外与可见光图像进行融合。");
  promptInput.focus();
}

$("sendSmartBtn").addEventListener("click", smartSend);
$("generateReportBtn").addEventListener("click", generateCurrentReport);
$("loadCurrentFusionBtn").addEventListener("click", loadCurrentFusion);
$("newSessionBtn").addEventListener("click", newSession);
$("clearSessionBtn").addEventListener("click", clearSession);
$("restoreSessionBtn").addEventListener("click", () => restoreSession(sessionId));
$("sessionSearch")?.addEventListener("input", renderSessionList);
$("batchPairMode")?.addEventListener("change", renderBatchPreview);
$("batchPreviewBtn")?.addEventListener("click", renderBatchPreview);
$("navChat").addEventListener("click", focusChat);
$("clearFlowBtn").addEventListener("click", clearWorkflow);
$("toggleWorkflowBtn")?.addEventListener("click", () => {
  setWorkflowExpanded($("appShell").classList.contains("workflow-collapsed"));
});
$("closeWorkflowBtn")?.addEventListener("click", () => setWorkflowExpanded(false));

chatArea.addEventListener("click", (event) => {
  const starter = event.target.closest(".starter-prompt");
  if (!starter) return;
  promptInput.value = starter.dataset.prompt || "";
  promptInput.focus();
});

promptInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) smartSend();
});

mergeSessionRows([]);
renderSessionList();
setWorkflowExpanded(localStorage.getItem("mambadfuse_workflow_expanded") !== "false");
fetchSystemInfo();
bootstrapSessionUI();
