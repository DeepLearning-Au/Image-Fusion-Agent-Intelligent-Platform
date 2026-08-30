const $ = (id) => document.getElementById(id);

function esc(value = "") {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function escAttr(value = "") {
  return esc(value).replaceAll('"', "&quot;").replaceAll("'", "&#39;");
}

async function api(url, opts = {}) {
  const response = await fetch(url, opts);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.message || data.detail || JSON.stringify(data) || response.statusText);
  }
  return data;
}

function statusFrom(data) {
  return data.status || data || {};
}

function docIdOf(doc) {
  return doc.doc_id ?? doc.id;
}

function chunkTextOf(chunk) {
  return chunk.text ?? chunk.content ?? "";
}

function chunkIdOf(chunk) {
  return chunk.chunk_id ?? chunk.id ?? "";
}

function formatSpecValue(spec) {
  let value = spec.value;
  if (value === true) value = "支持";
  else if (value === false) value = "不支持";
  else if (value == null) value = "资料未说明";
  else if (Array.isArray(value)) value = spec.comparator === "range" && value.length === 2 ? value.join("~") : value.join("、");
  const comparator = { lte: "≤", lt: "<", gte: "≥", gt: ">" }[spec.comparator] || "";
  return `${comparator}${value}${spec.unit || ""}`;
}

function setBox(id, html) {
  $(id).innerHTML = html;
}

async function runAction(action, messageBox = null) {
  try {
    return await action();
  } catch (err) {
    const msg = esc(err.message || err);
    if (messageBox) setBox(messageBox, `<div class="empty">操作失败：${msg}</div>`);
    alert(`操作失败：${err.message || err}`);
    throw err;
  }
}

async function loadStatus() {
  const [data, productData, candidateData] = await Promise.all([
    api("/api/kb/status"),
    api("/api/products/status"),
    api("/api/product-candidates/status"),
  ]);
  const s = statusFrom(data);
  setBox(
    "statusCards",
    [
      ["文档数", s.doc_count ?? 0],
      ["Chunk数", s.chunk_count ?? 0],
      ["结构化产品", productData.product_count ?? 0],
      ["结构化参数", productData.spec_value_count ?? 0],
      ["待审核参数", candidateData.pending_count ?? 0],
      ["待OCR页", s.needs_ocr_page_count ?? 0],
      ["解析警告", s.parse_warning_count ?? 0],
      ["识别表格", s.table_count ?? 0],
      ["提取图片", s.extracted_image_count ?? 0],
      ["PDF解析器", s.pdf_parser?.available ? "可用" : "未安装"],
      ["FTS5", s.fts5_available ? "可用" : "LIKE兜底"],
    ]
      .map(([k, v]) => `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`)
      .join("")
  );
}

function candidateValue(candidate) {
  return typeof candidate.value === "string" ? candidate.value : JSON.stringify(candidate.value);
}

async function loadCandidates() {
  const status = $("candidateStatus").value;
  const model = $("candidateModel").value.trim();
  const params = new URLSearchParams();
  if (status) params.set("status", status);
  if (model) params.set("model", model);
  const data = await api(`/api/product-candidates?${params.toString()}`);
  const candidates = data.candidates || [];
  if (!candidates.length) {
    setBox("candidateList", '<div class="empty">当前筛选条件下没有候选参数。</div>');
    return;
  }
  setBox("candidateList", candidates.map((candidate) => `<div class="candidate" data-id="${candidate.id}">
    <div class="candidate-head">
      <strong>${esc(candidate.model)} · ${esc(candidate.spec_label)}</strong>
      <span class="candidate-status ${escAttr(candidate.review_status)}">${esc(candidate.review_status)}</span>
    </div>
    <div class="meta">来源：${esc(candidate.source_document_title)}，第 ${candidate.source_page} 页；置信度：${candidate.confidence}</div>
    <div class="candidate-grid">
      <label>字段键<input class="candidate-key" value="${escAttr(candidate.spec_key)}" /></label>
      <label>类型<select class="candidate-type">
        ${["text", "number", "boolean", "json"].map((type) => `<option value="${type}" ${candidate.value_type === type ? "selected" : ""}>${type}</option>`).join("")}
      </select></label>
      <label>值<input class="candidate-value" value="${escAttr(candidateValue(candidate))}" /></label>
      <label>单位<input class="candidate-unit" value="${escAttr(candidate.unit || "")}" /></label>
    </div>
    <div class="meta">原文：${esc(candidate.raw_value)}</div>
    <div class="doc-actions">
      <button class="small candidate-approve" data-id="${candidate.id}">通过</button>
      <button class="small danger candidate-reject" data-id="${candidate.id}">拒绝</button>
    </div>
  </div>`).join(""));
  document.querySelectorAll(".candidate-approve").forEach((button) => {
    button.onclick = () => reviewCandidate(Number(button.dataset.id), "approved");
  });
  document.querySelectorAll(".candidate-reject").forEach((button) => {
    button.onclick = () => reviewCandidate(Number(button.dataset.id), "rejected");
  });
}

async function loadPublishableModels() {
  const data = await api("/api/product-candidates?status=approved");
  const models = [...new Set((data.candidates || []).map((item) => item.model).filter(Boolean))];
  const select = $("publishModel");
  const previous = select.value;
  select.innerHTML = '<option value="">选择已通过型号</option>' + models
    .map((model) => `<option value="${escAttr(model)}">${esc(model)}</option>`)
    .join("");
  if (models.includes(previous)) select.value = previous;
  $("publishCandidatesBtn").disabled = models.length === 0;
}

function readCandidateValue(element, valueType) {
  const raw = element.value.trim();
  if (valueType === "number") {
    const value = Number(raw);
    if (!Number.isFinite(value)) throw new Error("number类型必须填写数字");
    return value;
  }
  if (valueType === "boolean") {
    if (["true", "支持", "1"].includes(raw.toLowerCase())) return true;
    if (["false", "不支持", "0"].includes(raw.toLowerCase())) return false;
    throw new Error("boolean类型请填写true/false或支持/不支持");
  }
  if (valueType === "json") return JSON.parse(raw);
  return raw;
}

async function reviewCandidate(id, reviewStatus) {
  const box = document.querySelector(`.candidate[data-id="${id}"]`);
  const valueType = box.querySelector(".candidate-type").value;
  await runAction(async () => {
    await api(`/api/product-candidates/${id}/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        review_status: reviewStatus,
        spec_key: box.querySelector(".candidate-key").value.trim(),
        value_type: valueType,
        value: readCandidateValue(box.querySelector(".candidate-value"), valueType),
        unit: box.querySelector(".candidate-unit").value.trim() || null,
      }),
    });
    await Promise.all([loadCandidates(), loadPublishableModels(), loadStatus()]);
  }, "candidateList");
}

async function loadProducts() {
  const data = await api("/api/kb/product-assets");
  const products = data.products || [];
  $("catalogVersion").textContent = data.catalog_version ? `图谱 v${data.catalog_version}` : "";
  $("productDisclaimer").textContent = data.disclaimer || "";
  if (!products.length) {
    setBox("productGallery", '<div class="empty">暂无产品图谱资产。</div>');
    return;
  }
  setBox(
    "productGallery",
    products.map((product) => {
      const tags = (product.tags || []).map((tag) => `<span class="tag">${esc(tag)}</span>`).join("");
      const highlights = (product.highlights || []).map((item) => `<li>${esc(item)}</li>`).join("");
      const limitations = (product.limitations || []).map((item) => `<li>${esc(item)}</li>`).join("");
      const specs = (product.specs || []).slice(0, 10).map((spec) =>
        `<tr><th>${esc(spec.label || spec.spec_key)}</th><td>${esc(formatSpecValue(spec))}</td><td>第 ${esc(spec.source_page || "?")} 页</td></tr>`
      ).join("");
      return `<article class="product-card">
        <div class="product-image-wrap">
          <img class="product-image" src="${esc(product.image_url || "")}" alt="${esc(product.name || "产品示意图")}" loading="lazy" />
          <span class="illustration-badge">官方资料页</span>
        </div>
        <div class="product-body">
          <div class="product-category">${esc(product.category || "")}</div>
          <h4>${esc(product.name || "未命名产品")}</h4>
          <div class="meta">厂商：${esc(product.manufacturer || "资料未说明")}；型号：${esc(product.model || "")}</div>
          <p>${esc(product.description || "")}</p>
          <p><strong>价格：</strong>${esc(product.price_display || "价格未知")}</p>
          <div class="tags">${tags}</div>
          <details>
            <summary>能力与边界</summary>
            <div class="detail-grid">
              <div><strong>典型能力</strong><ul>${highlights}</ul></div>
              <div><strong>使用边界</strong><ul>${limitations}</ul></div>
            </div>
          </details>
          <details>
            <summary>结构化参数与来源</summary>
            <div class="table-wrap"><table><tbody>${specs}</tbody></table></div>
            <p class="meta">主要来源：${esc(product.source_document || "")}，第 ${esc(product.source_page || "?")} 页</p>
          </details>
          <button class="small secondary product-search" data-query="${esc(product.name || "")}">检索相关知识</button>
        </div>
      </article>`;
    }).join("")
  );
  document.querySelectorAll(".product-search").forEach((button) => {
    button.addEventListener("click", () => {
      $("searchInput").value = button.dataset.query || "";
      $("searchBtn").click();
      $("searchResult").scrollIntoView({ behavior: "smooth", block: "center" });
    });
  });
}

function scrollProductGallery(direction) {
  const gallery = $("productGallery");
  const distance = Math.min(gallery.clientWidth * 0.82, 460);
  gallery.scrollBy({ left: direction * distance, behavior: "smooth" });
}

async function loadDocs() {
  const keyword = $("docKeyword").value || "";
  const data = await api(`/api/kb/documents?keyword=${encodeURIComponent(keyword)}&include_disabled=true`);
  const docs = data.documents || [];
  if (!docs.length) {
    setBox("docList", '<div class="empty">暂无文档。可以先上传文档，或点击“从 data/knowledge_docs 重建”。</div>');
    return;
  }

  setBox(
    "docList",
    docs
      .map((doc) => {
        const id = docIdOf(doc);
        const enabled = Boolean(doc.enabled);
        const parseStatus = doc.parse_status || "未记录";
        const parser = doc.parser_name || "旧版读取器";
        const pageInfo = doc.page_count ? `；页数：${doc.page_count}；原生解析：${doc.native_page_count ?? 0}；OCR：${doc.ocr_page_count ?? 0}；待OCR：${doc.needs_ocr_page_count ?? 0}；表格：${doc.table_count ?? 0}；图片：${doc.extracted_image_count ?? 0}` : "";
        return `<div class="doc">
          <div class="title">${esc(doc.title || "未命名文档")}</div>
          <div class="meta">ID: ${esc(id)}；类别：${esc(doc.category || "")}；Chunk：${doc.chunk_count ?? 0}；状态：${enabled ? "启用" : "禁用"}</div>
          <div class="meta">解析器：${esc(parser)}；解析状态：${esc(parseStatus)}${pageInfo}</div>
          <div class="doc-actions">
            <button class="small" onclick="showChunks(${Number(id)})">查看内容</button>
            <button class="small secondary" onclick="showParseReport(${Number(id)})">解析报告</button>
            <button class="small secondary" onclick="toggleDoc(${Number(id)}, ${enabled ? "false" : "true"})">${enabled ? "禁用" : "启用"}</button>
            <button class="small danger" onclick="deleteDoc(${Number(id)})">删除</button>
          </div>
        </div>`;
      })
      .join("")
  );
}

async function showParseReport(docId) {
  await runAction(async () => {
    const data = await api(`/api/kb/documents/${encodeURIComponent(docId)}/parse-report`);
    const report = data.report || {};
    const pages = report.pages || [];
    const summary = `<div class="chunk">
      <div class="title">${esc(report.title || "文档")} - PDF 解析报告</div>
      <div class="meta">解析器：${esc(report.parser_name || "未记录")} ${esc(report.parser_version || "")}；状态：${esc(report.parse_status || "未记录")}</div>
      <pre>总页数：${report.page_count ?? 0}\n原生解析页：${report.native_page_count ?? 0}\nOCR页：${report.ocr_page_count ?? 0}\n待OCR页：${report.needs_ocr_page_count ?? 0}\n表格：${report.table_count ?? 0}\n提取图片：${report.extracted_image_count ?? 0}\n警告数：${report.parse_warning_count ?? 0}</pre>
    </div>`;
    const detail = pages.map((page) => `<div class="chunk">
      <div class="title">第 ${page.page_number} 页</div>
      <div class="meta">方式：${esc(page.parse_method || "-")}；状态：${esc(page.status || "-")}；质量分：${page.quality_score ?? 0}；表格：${page.table_count ?? 0}；提取图片：${page.extracted_image_count ?? 0}</div>
      <pre>有效字符：${page.meaningful_char_count ?? 0}\n乱码比例：${page.garbled_ratio ?? 0}\n是否待OCR：${page.needs_ocr ? "是" : "否"}${page.warning ? `\n说明：${esc(page.warning)}` : ""}</pre>
    </div>`).join("");
    setBox("chunkList", summary + detail);
    $("chunkList").scrollIntoView({ behavior: "smooth", block: "start" });
  }, "chunkList");
}

async function showChunks(docId) {
  await runAction(async () => {
    const data = await api(`/api/kb/chunks?doc_id=${encodeURIComponent(docId)}&limit=200`);
    const chunks = data.chunks || [];
    setBox(
      "chunkList",
      chunks
        .map(
          (chunk) => `<div class="chunk">
            <div class="title">${esc(chunk.title || chunk.doc_title || "文档")} #${chunk.chunk_index ?? ""}</div>
            <div class="meta">${esc(chunkIdOf(chunk))}</div>
            <pre>${esc(chunkTextOf(chunk))}</pre>
          </div>`
        )
        .join("") || '<div class="empty">这个文档暂无 Chunk 内容。</div>'
    );
  }, "chunkList");
}

async function deleteDoc(id) {
  if (!confirm("确认删除这个文档及其所有 chunks？")) return;
  await runAction(async () => {
    await api(`/api/kb/documents/${encodeURIComponent(id)}`, { method: "DELETE" });
    await loadStatus();
    await loadDocs();
    setBox("chunkList", '点击某个文档的“查看内容”后显示。');
  });
}

async function toggleDoc(id, enabled) {
  await runAction(async () => {
    await api(`/api/kb/documents/${encodeURIComponent(id)}/toggle`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled }),
    });
    await loadStatus();
    await loadDocs();
  });
}

async function loadAll() {
  await runAction(async () => {
    await Promise.all([loadStatus(), loadProducts(), loadDocs(), loadCandidates(), loadPublishableModels()]);
  }, "docList");
}

$("candidateStatus").onchange = () => loadCandidates().catch((err) => setBox("candidateList", `<div class="empty">加载失败：${esc(err.message)}</div>`));
$("candidateModel").oninput = () => loadCandidates().catch((err) => setBox("candidateList", `<div class="empty">加载失败：${esc(err.message)}</div>`));
$("generateCandidatesBtn").onclick = async () => {
  await runAction(async () => {
    const data = await api("/api/product-candidates/generate", { method: "POST" });
    await Promise.all([loadCandidates(), loadPublishableModels(), loadStatus()]);
    alert(`已读取 ${data.tables_seen} 个表格，生成/更新 ${data.candidates_seen} 个候选参数`);
  }, "candidateList");
};
$("publishCandidatesBtn").onclick = async () => {
  const model = $("publishModel").value;
  if (!model) {
    alert("当前没有已通过的型号，请先审核并通过参数");
    return;
  }
  await runAction(async () => {
    const data = await api("/api/product-candidates/publish", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model }),
    });
    await Promise.all([loadCandidates(), loadPublishableModels(), loadProducts(), loadStatus()]);
    alert(`已发布 ${data.published_count} 个参数`);
  }, "candidateList");
};

$("refreshBtn").onclick = loadAll;
$("productPrevBtn").onclick = () => scrollProductGallery(-1);
$("productNextBtn").onclick = () => scrollProductGallery(1);
$("productGallery").addEventListener("wheel", (event) => {
  const gallery = event.currentTarget;
  if (Math.abs(event.deltaY) <= Math.abs(event.deltaX)) return;
  const maxScroll = gallery.scrollWidth - gallery.clientWidth;
  const canMove = (event.deltaY < 0 && gallery.scrollLeft > 0) || (event.deltaY > 0 && gallery.scrollLeft < maxScroll - 1);
  if (!canMove) return;
  event.preventDefault();
  gallery.scrollLeft += event.deltaY;
}, { passive: false });
$("docKeyword").oninput = () => loadDocs().catch((err) => setBox("docList", `<div class="empty">加载失败：${esc(err.message)}</div>`));

$("addTextBtn").onclick = async () => {
  const title = $("addTitle").value.trim();
  const content = $("addContent").value.trim();
  if (!title || !content) {
    alert("标题和内容不能为空");
    return;
  }
  await runAction(async () => {
    await api("/api/kb/add-text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title,
        content,
        category: $("addCategory").value.trim() || "手动添加",
        enabled: true,
      }),
    });
    $("addContent").value = "";
    await loadStatus();
    await loadDocs();
    alert("已添加");
  });
};

$("uploadBtn").onclick = async () => {
  const files = $("uploadFiles").files;
  if (!files.length) {
    alert("请选择文件");
    return;
  }
  await runAction(async () => {
    const fd = new FormData();
    [...files].forEach((file) => fd.append("files", file));
    fd.append("category", $("uploadCategory").value.trim() || "上传文档");
    fd.append("enabled", "true");
    const data = await api("/api/kb/upload", { method: "POST", body: fd });
    await loadStatus();
    await loadDocs();
    const added = data.added || [];
    const errors = data.errors || [];
    const pendingOcr = added.reduce((total, item) => total + Number(item.parse_report?.needs_ocr_page_count || 0), 0);
    const tableCount = added.reduce((total, item) => total + Number(item.parse_report?.table_count || 0), 0);
    let candidateMessage = "";
    if (tableCount > 0) {
      await api("/api/product-candidates/generate", { method: "POST" });
      $("candidateStatus").value = "pending";
      await Promise.all([loadCandidates(), loadStatus()]);
      const pending = await api("/api/product-candidates?status=pending&model=");
      const models = [...new Set((pending.candidates || []).map((item) => item.model))];
      candidateMessage = models.length
        ? `\n已识别型号：${models.join("、")}。请在“产品参数审核”中审核并发布，之后才会出现在产品目录。`
        : "\n检测到参数表，但未识别出型号，请检查表头格式。";
      $("review")?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    alert(`上传完成：${added.length} 个，错误：${errors.length} 个，待OCR页：${pendingOcr}${candidateMessage}`);
  });
};

$("rebuildBtn").onclick = async () => {
  if (!confirm("将从 data/knowledge_docs 重建 SQLite 知识库，是否继续？")) return;
  await runAction(async () => {
    const data = await api("/api/kb/rebuild-folder", { method: "POST" });
    await loadStatus();
    await loadDocs();
    setBox("chunkList", '点击某个文档的“查看内容”后显示。');
    alert(`重建完成：扫描 ${data.files_seen ?? 0} 个文件，成功 ${(data.imported || []).length} 个，错误 ${(data.errors || []).length} 个`);
  });
};

$("clearBtn").onclick = async () => {
  if (!confirm("确认清空数据库？不会删除 knowledge_docs 原始文件。")) return;
  await runAction(async () => {
    await api("/api/kb/clear", { method: "POST" });
    await loadStatus();
    await loadDocs();
    setBox("chunkList", '点击某个文档的“查看内容”后显示。');
  });
};

$("searchBtn").onclick = async () => {
  const query = $("searchInput").value.trim();
  if (!query) {
    alert("请输入检索内容");
    return;
  }
  await runAction(async () => {
    const data = await api("/api/kb/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: 8 }),
    });
    const sources = data.sources || [];
    setBox(
      "searchResult",
      `<div class="meta">置信度：${esc(data.confidence || "low")}；最高分：${data.top_score ?? 0}；来源数：${sources.length}</div>` +
        (sources
          .map(
            (s) => `<div class="source">
              <div class="title">${esc(s.doc_title || s.title || "未知文档")}</div>
              <div class="meta">类别：${esc(s.category || "")}；得分：${s.score ?? s.final_score ?? 0}；方式：${esc(s.match_type || "local")}</div>
              <pre>${esc(s.highlight || s.text || s.content || "")}</pre>
            </div>`
          )
          .join("") || '<div class="empty">未命中来源</div>')
    );
  }, "searchResult");
};

loadAll();
