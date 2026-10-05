/**
 * ConSQL Next-Gen Real-Time Web Application Logic
 * Integrates real-time REST & SSE pipeline runner, Vis.js graph network,
 * Chart.js benchmark visuals, and live SQLite database execution.
 */

// Application State
const state = {
  databases: [],
  currentDbId: "concert_singer",
  currentDbData: null,
  activePipelineData: null,
  visNetwork: null,
  charts: {},
  pipelineLattice: null,
  navLattice: null,
  sqlLattice: null,
  playgroundLattice: null,
};

// Preset Example Queries
const EXAMPLES = [
  { q: "How many singers do we have?", db: "concert_singer", tag: "Aggregation" },
  { q: "List all singers and the concerts they performed in.", db: "concert_singer", tag: "Join Path" },
  { q: "How many heads of the departments are older than 56?", db: "department_management", tag: "Condition" },
  { q: "What departments are managed by temporary acting heads?", db: "department_management", tag: "Multi-Table" },
  { q: "What are the names of students enrolled in more than 2 courses?", db: "student_transcripts_tracking", tag: "Complex Join" },
  { q: "How many superheroes have a height above average?", db: "superhero", tag: "Subquery" },
  { q: "What is the average age of players in formula 1?", db: "formula_1", tag: "Math" },
  { q: "List all card games released after 2010.", db: "card_games", tag: "Filter" },
];

// Document Ready
document.addEventListener("DOMContentLoaded", () => {
  initNavbar();
  initExamples();
  initTabs();
  initLatticeLoaders();
  initLatticePlayground();
  fetchSystemStatus();
  fetchDatabases();
  initEventListeners();
  loadBenchmarks();
});

// -----------------------------------------------------------------------------
// Initialization & Navigation
// -----------------------------------------------------------------------------

function initNavbar() {
  const links = document.querySelectorAll(".nav-link");
  links.forEach(link => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const target = link.getAttribute("data-target");
      if (target) {
        switchTab(target);
      }
    });
  });
}

function initTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const tabId = btn.getAttribute("data-tab");
      switchTab(tabId);
    });
  });
}

function switchTab(tabId) {
  // Update Buttons
  document.querySelectorAll(".tab-btn").forEach(b => {
    b.classList.toggle("active", b.getAttribute("data-tab") === tabId);
  });
  // Update Panes
  document.querySelectorAll(".tab-content").forEach(content => {
    content.classList.toggle("active", content.id === `tab-${tabId}`);
  });

  // Re-fit network graph if switching to graph tab
  if (tabId === "graph" && state.visNetwork) {
    setTimeout(() => {
      state.visNetwork.fit();
    }, 200);
  }
}

function initExamples() {
  const container = document.getElementById("examplesContainer");
  if (!container) return;

  container.innerHTML = "";
  EXAMPLES.forEach(ex => {
    const pill = document.createElement("button");
    pill.className = "example-pill";
    pill.innerHTML = `<span>${escapeHtml(ex.q)}</span> <span class="db-tag">${escapeHtml(ex.db)}</span>`;
    pill.addEventListener("click", () => {
      document.getElementById("queryInput").value = ex.q;
      const dbSelect = document.getElementById("dbSelect");
      if (dbSelect) {
        let found = false;
        for (let i = 0; i < dbSelect.options.length; i++) {
          if (dbSelect.options[i].value.toLowerCase() === ex.db.toLowerCase()) {
            dbSelect.selectedIndex = i;
            found = true;
            break;
          }
        }
        if (!found) {
          const opt = document.createElement("option");
          opt.value = ex.db;
          opt.textContent = `${ex.db}`;
          opt.selected = true;
          dbSelect.appendChild(opt);
        }
        state.currentDbId = dbSelect.value;
      }
      showToast(`Loaded example for ${ex.db}`);
    });
    container.appendChild(pill);
  });
}

function initEventListeners() {
  document.getElementById("runPipelineBtn").addEventListener("click", runPipeline);
  document.getElementById("clearBtn").addEventListener("click", clearForm);
  document.getElementById("dbSelect").addEventListener("change", (e) => {
    state.currentDbId = e.target.value;
  });

  // Schema Explorer Modal trigger
  const inspectDbBtn = document.getElementById("inspectDbBtn");
  if (inspectDbBtn) {
    inspectDbBtn.addEventListener("click", () => {
      openSchemaModal(state.currentDbId);
    });
  }

  // Execute modified SQL button
  const execModifiedBtn = document.getElementById("execModifiedBtn");
  if (execModifiedBtn) {
    execModifiedBtn.addEventListener("click", executeModifiedSQL);
  }

  // Copy Buttons
  document.querySelectorAll(".btn-copy").forEach(btn => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-copy-target");
      const targetEl = document.getElementById(targetId);
      if (targetEl) {
        const text = targetEl.value || targetEl.innerText;
        navigator.clipboard.writeText(text).then(() => {
          showToast("Copied to clipboard!", "success");
        });
      }
    });
  });

  // Graph control buttons
  const fitBtn = document.getElementById("graphFitBtn");
  if (fitBtn) {
    fitBtn.addEventListener("click", () => {
      if (state.visNetwork) state.visNetwork.fit();
    });
  }
}

// -----------------------------------------------------------------------------
// API Calls
// -----------------------------------------------------------------------------

async function fetchSystemStatus() {
  try {
    const res = await fetch("/api/status");
    if (!res.ok) throw new Error("Failed to fetch status");
    const data = await res.json();

    document.getElementById("statusBackend").textContent = "Online";
    document.getElementById("statusDbCount").textContent = `${data.total_databases} Schemas`;
    document.getElementById("statusDevice").textContent = data.device.toUpperCase();

    // Populate model provider label
    const provSelect = document.getElementById("providerSelect");
    if (provSelect) {
      provSelect.value = data.default_provider || "mock";
    }
  } catch (err) {
    console.error("Status error:", err);
    document.getElementById("statusBackend").textContent = "Offline";
  }
}

async function fetchDatabases() {
  try {
    const res = await fetch("/api/databases");
    if (!res.ok) throw new Error("Failed to fetch databases");
    const dbs = await res.json();
    state.databases = dbs;

    const select = document.getElementById("dbSelect");
    select.innerHTML = "";

    // Group databases by Spider / BIRD
    const spiderGroup = document.createElement("optgroup");
    spiderGroup.label = "Spider Dataset";
    const birdGroup = document.createElement("optgroup");
    birdGroup.label = "BIRD Mini-Dev Dataset";

    dbs.forEach(d => {
      const opt = document.createElement("option");
      opt.value = d.db_id;
      opt.textContent = `${d.db_id} (${d.num_tables} tables)`;
      if (d.dataset === "Spider") {
        spiderGroup.appendChild(opt);
      } else {
        birdGroup.appendChild(opt);
      }
    });

    select.appendChild(spiderGroup);
    select.appendChild(birdGroup);

    if (dbs.length > 0) {
      select.value = dbs[0].db_id;
      state.currentDbId = dbs[0].db_id;
    }
  } catch (err) {
    console.error("Database list error:", err);
    showToast("Could not load databases list", "error");
  }
}

// -----------------------------------------------------------------------------
// Pipeline Execution
// -----------------------------------------------------------------------------

async function runPipeline() {
  const question = document.getElementById("queryInput").value.trim();
  const dbSelect = document.getElementById("dbSelect");
  let dbId = dbSelect ? dbSelect.value : "";
  if (!dbId && state.currentDbId) {
    dbId = state.currentDbId;
  }
  if (!dbId && state.databases && state.databases.length > 0) {
    dbId = state.databases[0].db_id;
  }
  if (!dbId) {
    dbId = "concert_singer";
  }
  const provider = document.getElementById("providerSelect").value;
  const useQce = document.getElementById("qceToggle").checked;

  if (!question) {
    showToast("Please enter a natural language question.", "error");
    document.getElementById("queryInput").focus();
    return;
  }

  const runBtn = document.getElementById("runPipelineBtn");
  runBtn.disabled = true;
  runBtn.innerHTML = `<span class="pulse-dot"></span> Optimizing Graph...`;

  // Activate LatticeLoader Banner
  const banner = document.getElementById("latticeBanner");
  if (banner) banner.style.display = "flex";
  if (state.pipelineLattice) {
    state.pipelineLattice.setStatus("working", "Optimizing Relational Context...");
  }

  resetStepper();
  setStepActive(1);

  try {
    const res = await fetch("/api/pipeline/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        db_id: dbId,
        provider,
        use_qce: useQce,
      }),
    });

    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || "Pipeline execution failed");
    }

    const data = await res.json();
    state.activePipelineData = data;

    // Transition LatticeLoader to Done
    if (state.pipelineLattice) {
      state.pipelineLattice.setStatus("done", "Optimized in");
    }

    // Animate stepper completion
    markAllStepsComplete(data.timings);

    // Populate all tabs
    renderSchemaLinkingTab(data.schema_linking, data.qce);
    renderGraphTab(data.graph, data.scoring.best_path, data.scoring.best_path_ids);
    renderPathsTab(data.path_generation);
    renderScoringTab(data.scoring);
    renderPromptTab(data.prompt);
    renderSQLAndExecutionTab(data.sql, data.execution, data.timings, dbId);

    showToast(`ConSQL finished in ${data.timings.total_pipeline_ms} ms!`, "success");
    switchTab("sql");
  } catch (err) {
    console.error("Pipeline run error:", err);
    if (state.pipelineLattice) {
      state.pipelineLattice.setStatus("error", "Failed after");
    }
    showToast(err.message, "error");
  } finally {
    runBtn.disabled = false;
    runBtn.innerHTML = `<span>⚡</span> <span>Run ConSQL Pipeline</span>`;
  }
}

// -----------------------------------------------------------------------------
// Stepper UI Management
// -----------------------------------------------------------------------------

function resetStepper() {
  document.querySelectorAll(".step-card").forEach(card => {
    card.classList.remove("active", "completed");
    const timeEl = card.querySelector(".step-time");
    if (timeEl) timeEl.textContent = "";
  });
}

function setStepActive(stepNum) {
  const card = document.getElementById(`step-${stepNum}`);
  if (card) {
    card.classList.add("active");
  }
}

function markAllStepsComplete(timings) {
  const timeMap = {
    1: `${timings.schema_linking_ms} ms`,
    2: `${timings.graph_building_ms} ms`,
    3: `${timings.path_generation_ms} ms`,
    4: `${timings.scoring_ms} ms`,
    5: "Optimal",
    6: `${timings.llm_generation_ms} ms`,
    7: `${timings.execution_ms} ms`,
  };

  for (let i = 1; i <= 7; i++) {
    const card = document.getElementById(`step-${i}`);
    if (card) {
      card.classList.remove("active");
      card.classList.add("completed");
      const timeEl = card.querySelector(".step-time");
      if (timeEl && timeMap[i]) {
        timeEl.textContent = timeMap[i];
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Tab Rendering
// -----------------------------------------------------------------------------

function renderSchemaLinkingTab(linking, qce) {
  const tableContainer = document.getElementById("relevantTablesList");
  const columnContainer = document.getElementById("relevantColumnsList");
  const qceBadge = document.getElementById("qceBadge");

  // QCE display
  if (qce && qceBadge) {
    qceBadge.style.display = "inline-flex";
    qceBadge.className = `score-badge score-${qce.complexity.toLowerCase() === 'simple' ? 'high' : (qce.complexity.toLowerCase() === 'medium' ? 'med' : 'low')}`;
    qceBadge.textContent = `QCE Complexity: ${qce.complexity} (Budget: ${qce.beam_budget} paths, Max Depth: ${qce.max_join_depth})`;
  }

  // Tables
  tableContainer.innerHTML = "";
  linking.relevant_tables.forEach(t => {
    const card = document.createElement("div");
    card.className = "item-card";
    const percent = Math.min(100, Math.round(t.score * 100));
    const scoreClass = t.score >= 0.7 ? "score-high" : (t.score >= 0.4 ? "score-med" : "score-low");
    const barColor = t.score >= 0.7 ? "#10b981" : (t.score >= 0.4 ? "#06b6d4" : "#94a3b8");

    card.innerHTML = `
      <div>
        <div class="item-name">${escapeHtml(t.table)}</div>
        <div class="meter-wrapper">
          <div class="meter-fill" style="width: ${percent}%; background: ${barColor}"></div>
        </div>
      </div>
      <span class="score-badge ${scoreClass}">${t.score.toFixed(3)}</span>
    `;
    tableContainer.appendChild(card);
  });

  // Columns
  columnContainer.innerHTML = "";
  linking.relevant_columns.forEach(c => {
    const card = document.createElement("div");
    card.className = "item-card";
    const scoreClass = c.score >= 0.7 ? "score-high" : (c.score >= 0.4 ? "score-med" : "score-low");

    card.innerHTML = `
      <div>
        <div class="item-name">${escapeHtml(c.table)}.<span style="color:var(--accent-cyan)">${escapeHtml(c.column)}</span></div>
      </div>
      <span class="score-badge ${scoreClass}">${c.score.toFixed(3)}</span>
    `;
    columnContainer.appendChild(card);
  });
}

function renderGraphTab(graphData, bestPathNames, bestPathIds) {
  const container = document.getElementById("schemaGraphNetwork");
  if (!container) return;

  const pathIdsSet = new Set(bestPathIds || []);
  const bestPathNamesSet = new Set(bestPathNames || []);

  const nodes = new vis.DataSet(
    graphData.nodes.map(n => {
      const isBest = bestPathNamesSet.has(n.label);
      let color = {
        background: isBest ? "#ef4444" : (n.weight >= 0.7 ? "#10b981" : (n.weight >= 0.4 ? "#06b6d4" : "#334155")),
        border: isBest ? "#fee2e2" : "#1e293b",
        highlight: { background: "#818cf8", border: "#ffffff" }
      };

      return {
        id: n.id,
        label: `${n.label}\nw=${n.weight}`,
        shape: "box",
        margin: 10,
        color: color,
        font: { color: "#ffffff", face: "Inter", size: 13, bold: true },
        shadow: isBest ? { enabled: true, color: "rgba(239, 68, 68, 0.6)", size: 15 } : { enabled: true, size: 5 },
      };
    })
  );

  const edges = new vis.DataSet(
    graphData.edges.map(e => {
      const isBestEdge = (
        bestPathNames.includes(e.from_name) &&
        bestPathNames.includes(e.to_name) &&
        Math.abs(bestPathNames.indexOf(e.from_name) - bestPathNames.indexOf(e.to_name)) === 1
      );

      return {
        from: e.from,
        to: e.to,
        label: e.fk_label || `w=${e.weight}`,
        arrows: "to",
        width: isBestEdge ? 4 : Math.max(1.5, e.weight * 3.5),
        color: {
          color: isBestEdge ? "#ef4444" : (e.weight >= 0.5 ? "#10b981" : "#64748b"),
          highlight: "#f43f5e",
        },
        font: { size: 10, color: "#94a3b8", strokeWidth: 0, align: "top" },
        smooth: { type: "curvedCW", roundness: 0.15 },
      };
    })
  );

  const options = {
    physics: {
      solver: "forceAtlas2Based",
      forceAtlas2Based: {
        gravitationalConstant: -70,
        centralGravity: 0.015,
        springLength: 130,
        springConstant: 0.08,
      },
      stabilization: { iterations: 150 },
    },
    interaction: { hover: true, tooltipDelay: 200, zoomView: true, dragView: true },
  };

  if (state.visNetwork) {
    state.visNetwork.destroy();
  }

  state.visNetwork = new vis.Network(container, { nodes, edges }, options);
}

function renderPathsTab(pathGen) {
  document.getElementById("pathCountValid").textContent = pathGen.valid_paths_count;
  document.getElementById("pathCountInvalid").textContent = pathGen.invalid_paths_count;
  document.getElementById("pathCountTotal").textContent = pathGen.total_paths;

  const container = document.getElementById("validPathsList");
  container.innerHTML = "";

  pathGen.valid_paths.forEach((p, idx) => {
    const card = document.createElement("div");
    card.className = "item-card";
    const route = p.join(" ➔ ");
    card.innerHTML = `
      <div class="path-route">
        <span style="color:var(--text-muted)">#${idx + 1}</span>
        <span>${escapeHtml(route)}</span>
      </div>
      <span class="db-tag">${p.length} tables</span>
    `;
    container.appendChild(card);
  });
}

function renderScoringTab(scoring) {
  const container = document.getElementById("scoredPathsList");
  container.innerHTML = "";

  scoring.scored_paths.forEach((sp, idx) => {
    const isBest = idx === 0;
    const card = document.createElement("div");
    card.className = `path-item ${isBest ? 'best-path' : ''}`;

    const route = sp.path.join(" ➔ ");
    card.innerHTML = `
      <div class="path-header">
        <div class="path-route">
          ${isBest ? '<span class="score-badge score-high" style="background:#ef4444; color:#fff">🏆 Best Join Strategy</span>' : `<span style="color:var(--text-muted)">Rank #${idx + 1}</span>`}
          <span style="font-weight:600">${escapeHtml(route)}</span>
        </div>
        <div class="path-score-pill">${sp.final_score.toFixed(4)}</div>
      </div>
      <div class="path-metrics">
        <span>Semantic S_sem: <strong style="color:var(--accent-emerald)">${sp.sem_score.toFixed(3)}</strong></span>
        <span>Structural S_struct: <strong style="color:var(--accent-cyan)">${sp.struct_score.toFixed(3)}</strong></span>
        <span>Penalty L(p): <strong style="color:var(--accent-amber)">${sp.length_penalty.toFixed(3)}</strong></span>
        <span>Joins: <strong>${sp.joins}</strong></span>
      </div>
    `;
    container.appendChild(card);
  });
}

function renderPromptTab(prompt) {
  const promptEl = document.getElementById("serializedPromptText");
  if (promptEl) {
    promptEl.textContent = prompt;
  }
}

function renderSQLAndExecutionTab(sqlData, execution, timings, dbId) {
  const sqlArea = document.getElementById("predictedSqlArea");
  if (sqlArea) {
    sqlArea.value = sqlData.predicted_sql;
  }

  // SQL validation tag
  const valTag = document.getElementById("sqlValidationTag");
  if (valTag) {
    if (sqlData.is_valid) {
      valTag.className = "score-badge score-high";
      valTag.textContent = "✓ SQLGlot AST Valid";
    } else {
      valTag.className = "score-badge score-low";
      valTag.textContent = `⚠ Warning: ${sqlData.validation_error || 'Check SQL'}`;
    }
  }

  // LLM generation note if mock fallback
  const noteEl = document.getElementById("sqlNote");
  if (noteEl) {
    if (sqlData.generation_note) {
      noteEl.textContent = sqlData.generation_note;
      noteEl.style.display = "block";
    } else {
      noteEl.style.display = "none";
    }
  }

  // Latency & Execution Stats
  document.getElementById("statTotalTime").textContent = `${timings.total_pipeline_ms} ms`;
  document.getElementById("statExecTime").textContent = `${execution.execution_time_ms} ms`;
  document.getElementById("statRowsCount").textContent = execution.total_rows !== undefined ? execution.total_rows : (execution.rows ? execution.rows.length : 0);
  document.getElementById("statTablesCount").textContent = state.activePipelineData.scoring.best_path.length;

  // Render Table
  renderResultsTable(execution);
}

function renderResultsTable(execResult) {
  const tableEl = document.getElementById("executionResultsTable");
  const emptyEl = document.getElementById("emptyResultsMessage");

  if (!execResult.executed || !execResult.success || !execResult.columns || execResult.columns.length === 0) {
    tableEl.style.display = "none";
    emptyEl.style.display = "block";
    emptyEl.textContent = execResult.error ? `Execution Note: ${execResult.error}` : "Query executed successfully (0 rows returned).";
    return;
  }

  tableEl.style.display = "table";
  emptyEl.style.display = "none";

  // Build header
  const thead = tableEl.querySelector("thead");
  thead.innerHTML = "";
  const htr = document.createElement("tr");
  execResult.columns.forEach(c => {
    const th = document.createElement("th");
    th.textContent = c;
    htr.appendChild(th);
  });
  thead.appendChild(htr);

  // Build body
  const tbody = tableEl.querySelector("tbody");
  tbody.innerHTML = "";
  execResult.rows.forEach(r => {
    const tr = document.createElement("tr");
    r.forEach(cell => {
      const td = document.createElement("td");
      td.textContent = cell !== null ? String(cell) : "NULL";
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

// -----------------------------------------------------------------------------
// Direct Modified SQL Execution
// -----------------------------------------------------------------------------

async function executeModifiedSQL() {
  const sql = document.getElementById("predictedSqlArea").value.trim();
  const dbId = state.currentDbId;

  if (!sql) {
    showToast("SQL cannot be empty", "error");
    return;
  }

  if (state.sqlLattice) {
    state.sqlLattice.update({ showTimer: true });
    state.sqlLattice.setStatus("working", "Executing Query");
  }

  try {
    const res = await fetch("/api/execute_sql", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sql, db_id: dbId }),
    });

    if (!res.ok) throw new Error("Execution failed");
    const data = await res.json();

    document.getElementById("statExecTime").textContent = `${data.execution_time_ms} ms`;
    document.getElementById("statRowsCount").textContent = data.total_rows;
    renderResultsTable(data);

    if (data.success) {
      if (state.sqlLattice) state.sqlLattice.setStatus("done", "Executed in");
      showToast(`Query executed: ${data.total_rows} rows in ${data.execution_time_ms} ms`, "success");
    } else {
      if (state.sqlLattice) state.sqlLattice.setStatus("error", "Failed after");
      showToast(data.error || "Query failed to execute", "error");
    }
  } catch (err) {
    if (state.sqlLattice) state.sqlLattice.setStatus("error", "Failed after");
    showToast(err.message, "error");
  }
}

// -----------------------------------------------------------------------------
// Database Schema Inspector Modal
// -----------------------------------------------------------------------------

async function openSchemaModal(dbId) {
  const modal = document.getElementById("schemaModal");
  const modalTitle = document.getElementById("schemaModalTitle");
  const modalBody = document.getElementById("schemaModalBody");

  modalTitle.textContent = `Database Schema: ${dbId}`;
  modalBody.innerHTML = `<div style="text-align:center; padding:30px;"><span class="pulse-dot"></span> Loading tables and foreign keys...</div>`;
  modal.classList.add("active");

  try {
    const res = await fetch(`/api/databases/${dbId}/schema`);
    if (!res.ok) throw new Error("Failed to load schema");
    const data = await res.json();

    let html = `<div style="display:flex; flex-direction:column; gap:20px;">`;

    // Foreign Keys section
    if (data.foreign_keys && data.foreign_keys.length > 0) {
      html += `
        <div>
          <h4 style="font-size:14px; text-transform:uppercase; color:var(--accent-cyan); margin-bottom:10px;">Foreign Key Relationships (${data.foreign_keys.length})</h4>
          <div style="display:flex; flex-wrap:wrap; gap:8px;">
      `;
      data.foreign_keys.forEach(fk => {
        html += `<span class="example-pill"><strong style="color:#fff">${escapeHtml(fk.from_table)}.${escapeHtml(fk.from_col)}</strong> ➔ <span>${escapeHtml(fk.to_table)}.${escapeHtml(fk.to_col)}</span></span>`;
      });
      html += `</div></div>`;
    }

    // Tables & columns
    data.tables.forEach(t => {
      html += `
        <div style="background:rgba(15,23,42,0.8); border:1px solid var(--border-subtle); border-radius:var(--radius-md); padding:16px;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <h4 style="font-size:16px; color:#fff;">📊 ${escapeHtml(t.name)}</h4>
            <span class="db-tag">${t.columns.length} columns</span>
          </div>
          <div style="display:flex; flex-wrap:wrap; gap:6px; margin-bottom:12px;">
      `;

      t.columns.forEach(c => {
        const pkBadge = c.is_primary ? `<span style="color:var(--accent-amber); font-size:10px;">[PK]</span>` : '';
        html += `<span style="font-size:12px; background:rgba(255,255,255,0.05); padding:3px 8px; border-radius:4px; border:1px solid var(--border-subtle);">${escapeHtml(c.name)} ${pkBadge} <span style="color:var(--text-muted); font-size:10px;">(${c.type})</span></span>`;
      });
      html += `</div>`;

      // Sample rows
      if (t.sample_rows && t.sample_rows.length > 0) {
        html += `
          <div style="max-height:120px; overflow:auto; font-size:11px;">
            <table class="data-table">
              <thead><tr>${t.columns.map(c => `<th>${escapeHtml(c.name)}</th>`).join('')}</tr></thead>
              <tbody>${t.sample_rows.map(r => `<tr>${r.map(v => `<td>${escapeHtml(String(v))}</td>`).join('')}</tr>`).join('')}</tbody>
            </table>
          </div>
        `;
      }
      html += `</div>`;
    });

    html += `</div>`;
    modalBody.innerHTML = html;
  } catch (err) {
    modalBody.innerHTML = `<div style="color:var(--accent-rose); padding:20px;">Error: ${err.message}</div>`;
  }
}

// Close Modal
document.querySelectorAll(".modal-close-trigger").forEach(el => {
  el.addEventListener("click", () => {
    document.querySelectorAll(".modal-overlay").forEach(m => m.classList.remove("active"));
  });
});

// -----------------------------------------------------------------------------
// Benchmark Charts Visualizer (Chart.js)
// -----------------------------------------------------------------------------

async function loadBenchmarks() {
  try {
    const res = await fetch("/api/benchmarks");
    if (!res.ok) return;
    const data = await res.json();

    const spiderModels = data.datasets.spider.models;
    const birdModels = data.datasets.bird.models;

    // Spider Chart
    const ctxSpider = document.getElementById("spiderChart");
    if (ctxSpider) {
      new Chart(ctxSpider, {
        type: "bar",
        data: {
          labels: spiderModels.map(m => m.name),
          datasets: [
            {
              label: "Execution Accuracy (EX %)",
              data: spiderModels.map(m => m.ex),
              backgroundColor: spiderModels.map(m => m.highlight ? "rgba(99, 102, 241, 0.85)" : "rgba(148, 163, 184, 0.3)"),
              borderColor: spiderModels.map(m => m.highlight ? "#818cf8" : "rgba(148, 163, 184, 0.5)"),
              borderWidth: 1.5,
              borderRadius: 6,
            },
            {
              label: "Join Path F1 (%)",
              data: spiderModels.map(m => m.join_f1),
              backgroundColor: spiderModels.map(m => m.highlight ? "rgba(6, 182, 212, 0.85)" : "rgba(100, 116, 139, 0.3)"),
              borderColor: spiderModels.map(m => m.highlight ? "#22d3ee" : "rgba(100, 116, 139, 0.5)"),
              borderWidth: 1.5,
              borderRadius: 6,
            }
          ]
        },
        options: {
          responsive: true,
          plugins: { legend: { labels: { color: "#94a3b8" } } },
          scales: {
            y: { beginAtZero: true, max: 100, grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8" } },
            x: { grid: { display: false }, ticks: { color: "#94a3b8" } }
          }
        }
      });
    }

    // BIRD Mini-Dev Chart
    const ctxBird = document.getElementById("birdChart");
    if (ctxBird) {
      new Chart(ctxBird, {
        type: "bar",
        data: {
          labels: birdModels.map(m => m.name),
          datasets: [
            {
              label: "Execution Accuracy (EX %)",
              data: birdModels.map(m => m.ex),
              backgroundColor: birdModels.map(m => m.highlight ? "rgba(16, 185, 129, 0.85)" : "rgba(148, 163, 184, 0.3)"),
              borderColor: birdModels.map(m => m.highlight ? "#34d399" : "rgba(148, 163, 184, 0.5)"),
              borderWidth: 1.5,
              borderRadius: 6,
            },
            {
              label: "Join Path F1 (%)",
              data: birdModels.map(m => m.join_f1),
              backgroundColor: birdModels.map(m => m.highlight ? "rgba(245, 158, 11, 0.85)" : "rgba(100, 116, 139, 0.3)"),
              borderColor: birdModels.map(m => m.highlight ? "#fbbf24" : "rgba(100, 116, 139, 0.5)"),
              borderWidth: 1.5,
              borderRadius: 6,
            }
          ]
        },
        options: {
          responsive: true,
          plugins: { legend: { labels: { color: "#94a3b8" } } },
          scales: {
            y: { beginAtZero: true, max: 100, grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8" } },
            x: { grid: { display: false }, ticks: { color: "#94a3b8" } }
          }
        }
      });
    }
  } catch (err) {
    console.warn("Could not render benchmark charts:", err);
  }
}

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

function clearForm() {
  document.getElementById("queryInput").value = "";
  document.getElementById("predictedSqlArea").value = "";
  resetStepper();
  document.getElementById("relevantTablesList").innerHTML = "";
  document.getElementById("relevantColumnsList").innerHTML = "";
  document.getElementById("validPathsList").innerHTML = "";
  document.getElementById("scoredPathsList").innerHTML = "";
  document.getElementById("executionResultsTable").querySelector("tbody").innerHTML = "";
  document.getElementById("emptyResultsMessage").style.display = "block";
  document.getElementById("emptyResultsMessage").textContent = "Run a query to view live execution results.";
  showToast("Cleared inputs.");
}

function showToast(message, type = "info") {
  const container = document.getElementById("toastContainer");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;

  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(100%)";
    toast.style.transition = "all 0.3s ease";
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/[&<>"']/g, m => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[m]);
}

// -----------------------------------------------------------------------------
// LatticeLoader Initialization & Component Playground
// -----------------------------------------------------------------------------

function initLatticeLoaders() {
  if (typeof createLatticeLoader !== "function") {
    console.warn("createLatticeLoader is not loaded.");
    return;
  }

  // 1. Navbar Ambient Status Lattice
  const navMount = document.getElementById("navLatticeMount");
  if (navMount) {
    state.navLattice = createLatticeLoader(navMount, {
      status: "done",
      doneLabel: "Engine Ready",
      doneColor: "#00ffaa",
      color: "var(--accent-cyan)",
      cellSize: 4,
      gap: 1.5,
      fontSize: 12,
      showTimer: false,
      glow: true,
      pattern: "orbit",
      shape: "round"
    });
  }

  // 2. Real-Time Pipeline Execution Banner Lattice
  const pipeMount = document.getElementById("pipelineLatticeMount");
  if (pipeMount) {
    state.pipelineLattice = createLatticeLoader(pipeMount, {
      status: "working",
      label: "Optimizing Relational Context",
      doneLabel: "Relational Context Optimized in",
      errorLabel: "Optimization Failed after",
      pattern: "orbit",
      grid: 3,
      shape: "round",
      color: "var(--accent-cyan)",
      doneColor: "#00ffaa",
      errorColor: "#ff0055",
      cellSize: 7,
      gap: 2,
      fontSize: 14,
      step: 90,
      glow: true,
      glowColor: "var(--accent-cyan)",
      showTimer: true
    });
  }

  // 3. SQL Action Runner Lattice
  const sqlMount = document.getElementById("sqlLatticeMount");
  if (sqlMount) {
    state.sqlLattice = createLatticeLoader(sqlMount, {
      status: "done",
      doneLabel: "Ready",
      doneColor: "#00ffaa",
      errorColor: "#ff0055",
      color: "var(--text-secondary)",
      cellSize: 4,
      gap: 1.5,
      fontSize: 12,
      showTimer: false,
      glow: false,
      pattern: "dots"
    });
  }
}

function initLatticePlayground() {
  const mount = document.getElementById("playgroundLatticeMount");
  if (!mount || typeof createLatticeLoader !== "function") return;

  const statusSelect = document.getElementById("plgStatus");
  const patternSelect = document.getElementById("plgPattern");
  const gridSelect = document.getElementById("plgGrid");
  const shapeSelect = document.getElementById("plgShape");
  const labelInput = document.getElementById("plgLabel");
  const glowCheckbox = document.getElementById("plgGlow");
  const hintEl = document.getElementById("playgroundStatusHint");
  const codeArea = document.getElementById("plgCodeArea");

  let currentConfig = {
    status: statusSelect ? statusSelect.value : "working",
    label: labelInput ? labelInput.value : "Thinking",
    doneLabel: "Done in",
    errorLabel: "Failed after",
    pattern: patternSelect ? patternSelect.value : "orbit",
    grid: gridSelect ? parseInt(gridSelect.value, 10) : 3,
    shape: shapeSelect ? shapeSelect.value : "round",
    color: "#00f0ff",
    doneColor: "#00ffaa",
    errorColor: "#ff0055",
    cellSize: 8,
    gap: 2.5,
    fontSize: 15,
    step: 90,
    glow: glowCheckbox ? glowCheckbox.checked : true,
    glowColor: "#00f0ff",
    showTimer: true
  };

  mount.innerHTML = "";
  state.playgroundLattice = createLatticeLoader(mount, currentConfig);

  function updatePlayground() {
    currentConfig.status = statusSelect.value;
    currentConfig.pattern = patternSelect.value;
    currentConfig.grid = parseInt(gridSelect.value, 10);
    currentConfig.shape = shapeSelect.value;
    currentConfig.label = labelInput.value.trim() || "Thinking";
    currentConfig.glow = glowCheckbox.checked;

    if (state.playgroundLattice) {
      mount.innerHTML = "";
      state.playgroundLattice = createLatticeLoader(mount, currentConfig);
    }

    if (hintEl) {
      hintEl.textContent = `status: "${currentConfig.status}" | pattern: "${currentConfig.pattern}" (${currentConfig.grid}x${currentConfig.grid})`;
    }

    if (codeArea) {
      codeArea.textContent = `<LatticeLoader
  status="${currentConfig.status}"
  label="${currentConfig.label}"
  doneLabel="Done in"
  errorLabel="Failed after"
  pattern="${currentConfig.pattern}"
  grid={${currentConfig.grid}}
  shape="${currentConfig.shape}"
  color="${currentConfig.color}"
  doneColor="${currentConfig.doneColor}"
  errorColor="${currentConfig.errorColor}"
  cellSize={${currentConfig.cellSize}}
  gap={${currentConfig.gap}}
  fontSize={${currentConfig.fontSize}}
  step={${currentConfig.step}}
  glow={${currentConfig.glow}}
  showTimer={true}
/>`;
    }
  }

  [statusSelect, patternSelect, gridSelect, shapeSelect, labelInput, glowCheckbox].forEach(el => {
    if (el) el.addEventListener("change", updatePlayground);
  });
  if (labelInput) labelInput.addEventListener("input", updatePlayground);

  updatePlayground();
}
