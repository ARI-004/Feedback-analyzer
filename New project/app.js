const feedbackInput = document.querySelector("#feedback-text");
const fileInput = document.querySelector("#feedback-file");
const groupField = document.querySelector("#group-field");
const analyzeButton = document.querySelector("#analyze-feedback");
const loadSampleButton = document.querySelector("#load-sample");
const exportButton = document.querySelector("#export-report");
const cloudFilter = document.querySelector("#cloud-filter");
const wordCloud = document.querySelector("#word-cloud");
const ctx = wordCloud.getContext("2d");

const state = {
  rows: [],
  headers: [],
  report: null,
};

const stopWords = new Set([
  "a", "about", "after", "all", "also", "am", "an", "and", "are", "as", "at", "be",
  "because", "been", "but", "by", "can", "could", "did", "do", "does", "for", "from",
  "had", "has", "have", "he", "her", "his", "i", "if", "in", "into", "is", "it", "its",
  "just", "more", "my", "no", "not", "of", "on", "or", "our", "out", "over", "so",
  "some", "still", "than", "that", "the", "their", "them", "there", "this", "to",
  "too", "us", "was", "we", "were", "what", "when", "with", "would", "you", "your",
]);

const sampleFeedback = [
  "The rollout was smooth and the dashboard is much easier for regional teams to use.",
  "Support responses are still too slow, especially when a blocker is raised near a deadline.",
  "The training materials helped our team adopt the new process with confidence.",
  "Pricing feels high compared with the value smaller departments are getting right now.",
  "Communication improved this quarter, but we need clearer updates when timelines change.",
  "The reporting feature is useful and saves hours of manual work every week.",
  "Several users found the onboarding flow confusing and the documentation is missing examples.",
  "Quality is strong overall, although the export tool failed twice during review.",
  "Stakeholders appreciate the transparency in meetings and the quick follow-up notes.",
  "The mobile interface is hard to navigate and makes approvals slower than expected.",
].join("\n");

const tokenize = (text) =>
  text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, " ")
    .split(/\s+/)
    .map((word) => word.replace(/^-+|-+$/g, ""))
    .filter(Boolean);

const meaningfulTokens = (text) =>
  tokenize(text).filter((word) => word.length > 2 && !stopWords.has(word) && Number.isNaN(Number(word)));

const parseCsvLine = (line) => {
  const cells = [];
  let current = "";
  let quoted = false;

  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    const next = line[index + 1];

    if (char === '"' && next === '"') {
      current += '"';
      index += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      cells.push(current.trim());
      current = "";
    } else {
      current += char;
    }
  }

  cells.push(current.trim());
  return cells;
};

const detectCommentColumn = (headers) => {
  const preferred = ["comment", "feedback", "response", "text", "description", "survey response"];
  const lower = headers.map((header) => header.toLowerCase());
  const match = preferred.find((name) => lower.includes(name));
  return match ? lower.indexOf(match) : 0;
};

const parseInputRows = () => {
  if (state.rows.length) {
    return state.rows.map(({ comment, group }) => ({ comment, group }));
  }

  return feedbackInput.value
    .split(/\r?\n/)
    .map((comment) => ({ comment: comment.trim(), group: "" }))
    .filter((row) => row.comment);
};

const updateCommentCount = () => {
  const count = parseInputRows().length;
  document.querySelector("#comment-count").textContent = `${count} comment${count === 1 ? "" : "s"}`;
};

const percentage = (count, total) => (total ? `${Math.round((count / total) * 100)}%` : "0%");

const sentimentCounts = (items) =>
  items.reduce(
    (counts, item) => {
      counts[item.sentiment] = (counts[item.sentiment] || 0) + 1;
      return counts;
    },
    { positive: 0, neutral: 0, negative: 0, mixed: 0 },
  );

const topWords = (items, limit = 10) => {
  const frequencies = new Map();
  items.forEach((item) => {
    meaningfulTokens(item.comment).forEach((word) => {
      frequencies.set(word, (frequencies.get(word) || 0) + 1);
    });
  });
  return [...frequencies.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([word, count]) => ({ word, count }));
};

const currentItems = () => state.report?.items || [];

const renderMetrics = () => {
  const items = currentItems();
  const total = items.length;
  const counts = state.report?.sentiment || sentimentCounts(items);

  document.querySelector("#metric-total").textContent = total;
  document.querySelector("#metric-positive").textContent = percentage(counts.positive || 0, total);
  document.querySelector("#metric-neutral").textContent = percentage(counts.neutral || 0, total);
  document.querySelector("#metric-negative").textContent = percentage(counts.negative || 0, total);
  document.querySelector("#metric-mixed").textContent = percentage(counts.mixed || 0, total);
  document.querySelector("#confidence-note").textContent = total ? "ML model active" : "Ready";
};

const renderSummary = () => {
  const output = document.querySelector("#summary-output");
  output.textContent = state.report?.summary || "Add stakeholder feedback and run the ML analysis to generate sentiment, emotion, themes, summaries, and word clouds.";
};

const renderThemes = () => {
  const container = document.querySelector("#theme-list");
  const themes = state.report?.themes || [];

  if (!themes.length) {
    container.innerHTML = '<p class="empty-copy">No themes detected yet.</p>';
    return;
  }

  container.innerHTML = themes
    .map((theme) => {
      const tone = Object.entries({
        positive: theme.positive || 0,
        neutral: theme.neutral || 0,
        negative: theme.negative || 0,
        mixed: theme.mixed || 0,
      }).sort((a, b) => b[1] - a[1])[0][0];
      return `
        <article class="theme-item">
          <div>
            <strong>${escapeHtml(theme.theme)}</strong>
            <span>${theme.total} comment${theme.total === 1 ? "" : "s"} - mostly ${tone}</span>
          </div>
          <p>${escapeHtml(theme.examples[0] || "")}</p>
        </article>
      `;
    })
    .join("");
};

const renderBars = () => {
  const container = document.querySelector("#sentiment-bars");
  const items = currentItems();
  const total = items.length;
  const counts = state.report?.sentiment || sentimentCounts(items);
  container.innerHTML = ["positive", "neutral", "negative", "mixed"]
    .map((sentiment) => {
      const value = total ? Math.round(((counts[sentiment] || 0) / total) * 100) : 0;
      return `
        <div class="bar-row ${sentiment}">
          <div><span>${sentiment}</span><strong>${value}%</strong></div>
          <span class="bar-track"><span style="width:${value}%"></span></span>
        </div>
      `;
    })
    .join("");
};

const probabilityText = (probabilities) =>
  Object.entries(probabilities || {})
    .sort((a, b) => b[1] - a[1])
    .map(([label, value]) => `${label}: ${Math.round(value * 100)}%`)
    .join(", ");

const renderComments = () => {
  const list = document.querySelector("#comment-list");
  const items = currentItems();
  if (!items.length) {
    list.innerHTML = '<li class="empty-state">Run an analysis to see ML-classified comments.</li>';
    return;
  }

  list.innerHTML = items
    .map((item) => `
      <li class="comment-card ${item.sentiment}">
        <div class="comment-meta">
          <span>${escapeHtml(item.sentiment)} · ${escapeHtml(item.emotion || "emotion n/a")}</span>
          <span>${item.confidence ? `${Math.round(item.confidence * 100)}% confidence` : "scored"}</span>
        </div>
        <p>${escapeHtml(item.comment)}</p>
        <small>${item.themes.map(escapeHtml).join(", ")}${item.group ? ` - ${escapeHtml(item.group)}` : ""}</small>
        <small class="probability-line">Final: ${escapeHtml(item.sentimentSource || "model")} · model said ${escapeHtml(item.modelSentiment || item.sentiment)} class ${item.sentimentClass}: ${escapeHtml(probabilityText(item.sentimentProbabilities))}</small>
      </li>
    `)
    .join("");
};

const drawCloud = () => {
  const filter = cloudFilter.value;
  const items = filter === "all" ? currentItems() : currentItems().filter((item) => item.sentiment === filter);
  const words = topWords(items, 32);
  const width = wordCloud.width;
  const height = wordCloud.height;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(0, 0, width, height);

  if (!words.length) {
    ctx.fillStyle = "#64748b";
    ctx.font = "18px Arial";
    ctx.textAlign = "center";
    ctx.fillText("No words to display", width / 2, height / 2);
    return;
  }

  const max = Math.max(...words.map(({ count }) => count));
  const placed = [];
  const palette = ["#0f766e", "#1d4ed8", "#b45309", "#be123c", "#475569"];

  words.forEach(({ word, count }, index) => {
    const size = 18 + Math.round((count / max) * 34);
    ctx.font = `700 ${size}px Arial`;
    const metrics = ctx.measureText(word);
    const box = { width: metrics.width + 18, height: size + 14 };
    let x = 24 + ((index * 97) % Math.max(80, width - box.width - 48));
    let y = 44 + ((index * 61) % Math.max(80, height - box.height - 70));

    for (let attempt = 0; attempt < 80; attempt += 1) {
      const rect = { x, y: y - box.height, width: box.width, height: box.height };
      const overlaps = placed.some((other) =>
        rect.x < other.x + other.width &&
        rect.x + rect.width > other.x &&
        rect.y < other.y + other.height &&
        rect.y + rect.height > other.y,
      );
      if (!overlaps) {
        placed.push(rect);
        break;
      }
      x = 18 + ((x + 53) % Math.max(80, width - box.width - 36));
      y = 42 + ((y + 47) % Math.max(80, height - box.height - 64));
    }

    ctx.fillStyle = palette[index % palette.length];
    ctx.fillText(word, x, y);
  });
};

const renderAll = () => {
  renderMetrics();
  renderSummary();
  renderThemes();
  renderBars();
  renderComments();
  drawCloud();
};

const setBusy = (busy) => {
  analyzeButton.disabled = busy;
  analyzeButton.textContent = busy ? "Analyzing..." : "Analyze Feedback";
};

const runAnalysis = async () => {
  const rows = parseInputRows();
  setBusy(true);
  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rows }),
    });
    if (!response.ok) {
      throw new Error(`Analysis failed with status ${response.status}`);
    }
    state.report = await response.json();
    renderAll();
  } catch (error) {
    document.querySelector("#summary-output").textContent = error.message;
  } finally {
    setBusy(false);
  }
};

const escapeHtml = (value) =>
  String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  })[character]);

const populateGroupOptions = (headers) => {
  groupField.innerHTML = '<option value="">No group column</option>';
  headers.forEach((header) => {
    const option = document.createElement("option");
    option.value = header;
    option.textContent = `Group by ${header}`;
    groupField.append(option);
  });
};

const loadFile = async (file) => {
  const text = await file.text();
  state.rows = [];
  state.headers = [];

  if (file.name.toLowerCase().endsWith(".csv")) {
    const lines = text.split(/\r?\n/).filter((line) => line.trim());
    const headers = parseCsvLine(lines[0]);
    const commentIndex = detectCommentColumn(headers);
    const rows = lines.slice(1).map(parseCsvLine);
    state.headers = headers;
    state.rows = rows
      .map((cells) => ({
        comment: cells[commentIndex] || "",
        group: "",
        cells,
      }))
      .filter((row) => row.comment.trim());
    populateGroupOptions(headers.filter((_, index) => index !== commentIndex));
    feedbackInput.value = state.rows.map((row) => row.comment).join("\n");
  } else {
    feedbackInput.value = text;
    populateGroupOptions([]);
  }

  updateCommentCount();
};

groupField.addEventListener("change", () => {
  if (!state.rows.length || !groupField.value) return;
  const groupIndex = state.headers.indexOf(groupField.value);
  state.rows = state.rows.map((row) => ({ ...row, group: row.cells[groupIndex] || "" }));
});

feedbackInput.addEventListener("input", () => {
  state.rows = [];
  state.report = null;
  updateCommentCount();
});
fileInput.addEventListener("change", () => {
  const [file] = fileInput.files;
  if (file) loadFile(file);
});
analyzeButton.addEventListener("click", runAnalysis);
cloudFilter.addEventListener("change", drawCloud);
loadSampleButton.addEventListener("click", () => {
  state.rows = [];
  state.report = null;
  feedbackInput.value = sampleFeedback;
  populateGroupOptions([]);
  updateCommentCount();
  runAnalysis();
});
exportButton.addEventListener("click", () => {
  if (!state.report) return;
  const data = {
    generatedAt: new Date().toISOString(),
    ...state.report,
  };
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "feedback-ml-analysis-report.json";
  link.click();
  URL.revokeObjectURL(url);
});

updateCommentCount();
renderAll();
