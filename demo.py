# demo.py
# ============================================================
# ConSQL - Interactive Gradio Demo
# Shows every pipeline step visually for each user query
#
# Usage: python demo.py
# Then open: http://localhost:7860
# ============================================================

import sys, os, time
import socket
import logging
from pathlib import Path

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
for p in (CURRENT_DIR, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

import gradio as gr

from src.config import (
    SPIDER_DIR,
    BIRD_MINIDEV_DIR,
    SPIDER_DB_DIR,
    BIRD_DB_DIR,
    MAX_JOIN_DEPTH,
    LLM_PROVIDER,
    LLM_MODEL,
)
from src.data_loader import SpiderDataLoader
from src.bird_loader import BIRDMiniDevLoader
from src.retrieval.schema_linker import SBERTSchemaLinker
from src.graph.graph_builder import SchemaGraphBuilder
from src.graph.path_generator import PathGenerator
from src.graph.path_scorer import PathScorer
from src.llm.sql_generator import SQLGenerator
from src.llm.prompt_builder import PromptContext
from src.validator.sql_validator import SQLValidator
from src.executor.sql_executor import SQLExecutor
from src.pipeline.service import ConSQLPipeline

# Model & algorithm hyperparameters
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.3
ALPHA = 0.6
BETA = 0.4
DEVICE = "cpu"
MAX_PATHS = 50
SCORE_ALPHA = 0.4
SCORE_BETA = 0.4
SCORE_LAMBDA = 0.2

# -- Lazy-load pipeline (loaded once on startup) ---------------
_pipeline = None

def load_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    print("Loading ConSQL pipeline...")

    # Load schemas from both datasets
    all_schemas = {}

    if os.path.exists(SPIDER_DIR):
        try:
            spider_loader = SpiderDataLoader(spider_dir=str(SPIDER_DIR))
            spider_schemas = spider_loader.load_schemas()
            all_schemas.update(spider_schemas)
            print(f"  Spider: {len(spider_schemas)} schemas")
        except Exception as e:
            logger.warning(f"Could not load Spider schemas: {e}")

    if os.path.exists(BIRD_MINIDEV_DIR):
        try:
            bird_loader = BIRDMiniDevLoader(bird_minidev_dir=str(BIRD_MINIDEV_DIR))
            bird_schemas = bird_loader.load_schemas()
            all_schemas.update(bird_schemas)
            print(f"  BIRD:   {len(bird_schemas)} schemas")
        except Exception as e:
            logger.warning(f"Could not load BIRD schemas: {e}")

    # Build pipeline components
    linker = SBERTSchemaLinker(
        model_name=SBERT_MODEL,
        similarity_threshold=SIMILARITY_THRESHOLD,
        alpha=ALPHA, beta=BETA, device=DEVICE
    )
    builder = SchemaGraphBuilder(linker=linker)
    generator = PathGenerator(max_depth=MAX_JOIN_DEPTH, max_paths=MAX_PATHS)
    scorer = PathScorer(alpha=SCORE_ALPHA, beta=SCORE_BETA,
                        lam=SCORE_LAMBDA, max_depth=MAX_JOIN_DEPTH)
    sql_gen = SQLGenerator(provider=LLM_PROVIDER, model=LLM_MODEL)
    validator = SQLValidator()
    executor = SQLExecutor()

    pipeline = ConSQLPipeline(
        linker=linker,
        graph_builder=builder,
        path_generator=generator,
        path_scorer=scorer,
        sql_generator=sql_gen,
        validator=validator,
        executor=executor,
    )

    _pipeline = {
        "schemas":   all_schemas,
        "linker":    linker,
        "builder":   builder,
        "generator": generator,
        "scorer":    scorer,
        "sql_gen":   sql_gen,
        "validator": validator,
        "executor":  executor,
        "pipeline":  pipeline,
    }
    print("[OK] Pipeline loaded.")
    return _pipeline


# -- Core pipeline runner --------------------------------------

def run_pipeline(question: str, db_id: str):
    """
    Runs the full ConSQL pipeline for one query.
    Returns step-by-step outputs for display in the UI.
    """
    if not question.strip():
        return ("", "", None, "", "", "", "", "[FAIL] Please enter a question.", "")

    p = load_pipeline()
    schemas   = p["schemas"]
    linker    = p["linker"]
    builder   = p["builder"]
    generator = p["generator"]
    scorer    = p["scorer"]
    sql_gen   = p["sql_gen"]
    validator = p["validator"]
    executor  = p["executor"]

    schema = schemas.get(db_id)
    if schema is None:
        target = db_id.strip().lower() if db_id else ""
        for s_id, s in schemas.items():
            if s_id.lower() == target or target in s_id.lower() or s_id.lower() in target:
                db_id = s_id
                schema = s
                break
    if schema is None and schemas:
        db_id = "concert_singer" if "concert_singer" in schemas else next(iter(schemas.keys()))
        schema = schemas[db_id]

    if schema is None:
        return ("", "", None, "", "", "", "",
                f"[FAIL] Database '{db_id}' not found.\n"
                f"Available: {', '.join(list(schemas.keys())[:10])}...",
                "")

    timings = {}

    try:
        # -- Step 1: Schema Linking ----------------------------
        t0 = time.perf_counter()
        linking = linker.link(question, schema)
        timings["Schema Linking"] = time.perf_counter() - t0

        step1_out = f"  Relevant Tables ({len(linking.relevant_tables)}):\n"
        for ts in linking.relevant_tables:
            bar = " " * int(ts.score * 20)
            step1_out += f"  [{ts.score:.3f}] {ts.table.name_original:<25} {bar}\n"

        step1_out += f"\n  Relevant Columns ({len(linking.relevant_columns)}):\n"
        for cs in linking.relevant_columns[:8]:
            step1_out += (f"  [{cs.score:.3f}] "
                          f"{cs.table.name_original}.{cs.column.name_original}\n")

        # -- Step 2: Schema Graph ------------------------------
        t0 = time.perf_counter()
        sg_aware = builder.build(schema, linking_result=linking)
        timings["Graph Building"] = time.perf_counter() - t0

        edge_descriptions = []
        for t1, t2, data in sg_aware.graph.edges(data=True):
            n1 = sg_aware.table_id_to_name.get(t1, str(t1))
            n2 = sg_aware.table_id_to_name.get(t2, str(t2))
            w  = data.get("weight", 0)
            infos = sg_aware.edge_info.get((min(t1, t2), max(t1, t2)), [])
            fk_desc = ", ".join([f"{info.from_col_name}->{info.to_col_name}" for info in infos])
            edge_descriptions.append((n1, n2, w, fk_desc))

        logger.info(f"Generated nodes: {[(nid, sg_aware.table_id_to_name[nid], sg_aware.get_node_weight(nid)) for nid in sg_aware.graph.nodes()]}")
        logger.info(f"Generated edges with weights: {[ (e[0], e[1], e[2], e[3]) for e in edge_descriptions ]}")

        step2_out = f"  Schema Graph for [{db_id}]\n"
        step2_out += f"  Nodes (tables) : {sg_aware.graph.number_of_nodes()}\n"
        step2_out += f"  Edges (FK)     : {sg_aware.graph.number_of_edges()}\n\n"
        step2_out += "  FK Relationships:\n"
        for n1, n2, w, fk_desc in edge_descriptions:
            step2_out += f"  {n1:<22}    {n2:<22} (w={w:.3f}, FK={fk_desc})\n"

        # -- Step 3: Path Generation ---------------------------
        t0 = time.perf_counter()
        path_result = generator.generate(sg_aware, linking, schema)
        timings["Path Generation"] = time.perf_counter() - t0

        step3_out = (f"  Generated {len(path_result.all_paths)} total paths\n"
                     f"  [OK] Valid   : {path_result.num_valid}\n"
                     f"  [FAIL] Invalid : {path_result.num_invalid}\n\n"
                     f"  All Valid Paths:\n")
        for i, path in enumerate(path_result.valid_paths[:12]):
            pstr = " -> ".join(path.path_table_names)
            step3_out += f"  [{i+1:>2}] {pstr}\n"
        if path_result.num_valid > 12:
            step3_out += f"  ... and {path_result.num_valid - 12} more\n"

        # -- Step 4: Constraint Scoring ------------------------
        t0 = time.perf_counter()
        sel = scorer.score_and_select(path_result, sg_aware, linking)
        timings["Scoring"] = time.perf_counter() - t0

        # Runtime guard: don't let a missing best path crash the plot step
        if sel.best_path is None:
            raise RuntimeError("No valid join path found.")

        step4_out = (f"  Constraint Scoring\n"
                     f"  Formula: Score =   S_sem +   S_struct     L(p)\n"
                     f"  Weights:  ={SCORE_ALPHA},  ={SCORE_BETA},  ={SCORE_LAMBDA}\n\n"
                     f"  {'#':<4} {'Score':>6}  {'Sem':>6}  {'Struct':>6}  {'Pen':>6}  Path\n"
                     f"  {'-'*70}\n")
        for i, sp in enumerate(sel.scored_paths[:8]):
            pstr   = " -> ".join(sp.path.path_table_names)
            marker = " " if i == 0 else "  "
            step4_out += (f"  {marker}[{i+1}] {sp.final_score:>6.3f}  "
                          f"{sp.sem_score:>6.3f}  "
                          f"{sp.struct_score:>6.3f}  "
                          f"{sp.length_penalty:>6.3f}  {pstr}\n")

        # -- Step 5: Best Path ---------------------------------
        bp = sel.best_path
        best_path_names = bp.path.path_table_names
        logger.info(f"Final best join path: {[name for name in best_path_names]}")
        step5_out = (f"  Best Join Path Selected:\n\n"
                     f"  {' -> '.join(best_path_names)}\n\n"
                     f"  Score Breakdown:\n"
                     f"        S_sem     = {SCORE_ALPHA}   {bp.sem_score:.4f} = "
                     f"{SCORE_ALPHA * bp.sem_score:.4f}\n"
                     f"        S_struct  = {SCORE_BETA}   {bp.struct_score:.4f} = "
                     f"{SCORE_BETA * bp.struct_score:.4f}\n"
                     f"        L(p)      = {SCORE_LAMBDA}   {bp.length_penalty:.4f} = "
                     f"{SCORE_LAMBDA * bp.length_penalty:.4f}\n"
                     f"    -----------------------------\n"
                     f"    Final Score   = {bp.final_score:.4f}\n\n"
                     f"  Tables in join: {len(best_path_names)}\n"
                     f"  Join operations: {bp.path.num_joins}")

        # -- Step 6: Context Prompt ----------------------------
        tables_in_join = best_path_names if best_path_names else [ts.table.name_original for ts in linking.relevant_tables]
        columns_in_prompt = [f"{cs.table.name_original}.{cs.column.name_original}" for cs in linking.relevant_columns[:8]]
        prompt_ctx = PromptContext(
            question=question,
            db_id=db_id,
            tables=tables_in_join,
            columns=columns_in_prompt,
            join_path=best_path_names,
            foreign_keys=[f"{e[0]}.{e[3]} -> {e[1]}" for e in edge_descriptions[:5]],
        )
        prompt = sql_gen.prompt_builder.build(prompt_ctx)
        step6_out = f"  Context Prompt:\n\n{prompt}"

        # -- Step 7: SQL Generation ----------------------------
        t0 = time.perf_counter()
        try:
            gen_res = sql_gen.generate(
                question=question,
                db_id=db_id,
                tables=tables_in_join,
                columns=columns_in_prompt,
                join_path=best_path_names,
                foreign_keys=[],
            )
            sql_out = gen_res.predicted_sql
        except Exception as e:
            logger.warning(f"Default LLM generation failed ({e}), falling back to heuristic preview.")
            from src.llm.mock_client import MockLLMClient
            mock_client = MockLLMClient()
            sql_out = mock_client.generate_sql(prompt)
            sql_out = f"-- [Note: LLM provider unavailable ({e}); showing pipeline candidate preview]\n{sql_out}"
        timings["LLM Generation"] = time.perf_counter() - t0

        val_res = validator.validate(sql_out, schema)
        valid_tag = "[OK] Valid SQL" if val_res.is_valid else f"[WARN] Check SQL: {val_res.error}"

        db_path = os.path.join(SPIDER_DB_DIR, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            db_path = os.path.join(BIRD_DB_DIR, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            db_path = os.path.join(SPIDER_DB_DIR, "concert_singer", "concert_singer.sqlite")

        exec_tag = ""
        if os.path.exists(db_path):
            # Signature is execute(database_path: str, sql: str)
            exec_res = executor.execute(db_path, sql_out)
            if exec_res.success:
                exec_tag = f"\n[OK] Executed successfully: {len(exec_res.rows)} rows returned"
            else:
                exec_tag = f"\n[WARN] Execution note: {exec_res.error}"

        timings["Total"] = sum(timings.values())
        timing_lines = "\n".join(
            f"  {name:<16}: {seconds * 1000:.1f} ms" for name, seconds in timings.items()
        )
        step7_out = f"{sql_out}\n\n{valid_tag}{exec_tag}\n\n   Timing:\n{timing_lines}"

        # -- Plot the graph AFTER scoring, so the best path is known --
        schema_graph_fig = build_query_graph_figure(sg_aware, linking, bp)

        return (step1_out, step2_out, schema_graph_fig, step3_out,
                step4_out, step5_out, step6_out, step7_out, sql_out)

    except Exception:
        import traceback
        return ("", "", None, "", "", "", "",
                f"[FAIL] Pipeline error:\n{traceback.format_exc()}", "")


def build_query_graph_figure(sg_aware, linking, best_path=None):
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except Exception:
        return None

    relevant_ids = set(linking.get_table_ids())
    if best_path is not None:
        if hasattr(best_path, 'path_table_ids'):
            path_ids = set(best_path.path_table_ids)
        elif hasattr(best_path, 'path') and hasattr(best_path.path, 'path_table_ids'):
            path_ids = set(best_path.path.path_table_ids)
        else:
            path_ids = set()
    else:
        path_ids = set()

    # Display only SBERT-selected candidate tables (S_r) and any tables explicitly in the best path
    display_ids = set(relevant_ids) | path_ids

    if not display_ids:
        return None

    subgraph = sg_aware.graph.subgraph(display_ids).copy()
    if subgraph.number_of_nodes() == 0:
        return None

    node_scores = {nid: sg_aware.get_node_weight(nid) for nid in subgraph.nodes()}

    node_colors = []
    labels = {}
    label_bboxes = {}
    for nid in subgraph.nodes():
        score = node_scores.get(nid, 0.0)
        table_name = sg_aware.table_id_to_name.get(nid, str(nid))
        labels[nid] = f"{table_name}\n{score:.3f}"
        if nid in path_ids:
            color = "#ef4444"   # Best path -> red
        elif score >= 0.75:
            color = "#15803d"   # Highly relevant -> green
        elif score >= 0.4:
            color = "#38bdf8"   # Medium -> blue
        else:
            color = "#94a3b8"   # Low -> gray
        node_colors.append(color)
        label_bboxes[nid] = dict(boxstyle="round,pad=0.3", fc=color, ec="#111827", lw=1.5, alpha=0.95)

    fig, ax = plt.subplots(figsize=(14, 10), dpi=180)
    ax.set_title("Query-Specific Schema Graph", fontsize=18, fontweight='bold')
    ax.text(0.5, 0.94, "Best path highlighted, node score = SBERT relevance",
            transform=ax.transAxes, ha='center', fontsize=11, color='#475569')
    ax.text(0.02, 0.02, f"DB: {sg_aware.db_id}   |   Query: {linking.query}",
            transform=ax.transAxes, ha='left', fontsize=9, color='#475569')
    ax.axis("off")

    edge_weights = []
    edge_colors = []
    edge_labels = {}
    for u, v, data in subgraph.edges(data=True):
        weight = data.get("weight", 0.0)
        fk_label = data.get("fk", "")
        edge_weights.append(1.5 + 4.0 * weight)
        if weight >= 0.75:
            edge_colors.append("#064e3b")
        elif weight >= 0.4:
            edge_colors.append("#16a34a")
        else:
            edge_colors.append("#64748b")
        # Only the FK column mapping -- the weight is already encoded by edge thickness/color
        edge_labels[(u, v)] = fk_label if fk_label else f"{weight:.2f}"

    display_graph = nx.DiGraph()
    for nid, data in subgraph.nodes(data=True):
        display_graph.add_node(nid, **data)
    for u, v, data in subgraph.edges(data=True):
        direction = data.get("direction", (u, v))
        display_graph.add_edge(direction[0], direction[1], **data)

    pos = nx.kamada_kawai_layout(display_graph)

    nx.draw_networkx_edges(
        display_graph,
        pos,
        width=edge_weights,
        edge_color=edge_colors,
        alpha=0.85,
        arrows=True,
        arrowstyle='-|>',
        arrowsize=14,
        ax=ax
    )

    best_edge_list = []
    best_edge_widths = []
    if best_path is not None:
        if hasattr(best_path, 'path_table_ids'):
            bp_ids = best_path.path_table_ids
        elif hasattr(best_path, 'path') and hasattr(best_path.path, 'path_table_ids'):
            bp_ids = best_path.path.path_table_ids
        else:
            bp_ids = []
        for u, v in zip(bp_ids, bp_ids[1:]):
            if display_graph.has_edge(u, v):
                best_edge_list.append((u, v))
                best_edge_widths.append(5.5)
            elif display_graph.has_edge(v, u):
                best_edge_list.append((v, u))
                best_edge_widths.append(5.5)

    if best_edge_list:
        nx.draw_networkx_edges(
            display_graph,
            pos,
            edgelist=best_edge_list,
            width=best_edge_widths,
            edge_color="#ef4444",
            alpha=0.95,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=16,
            ax=ax
        )

    nx.draw_networkx_nodes(
        display_graph,
        pos,
        node_size=2000,
        node_color=node_colors,
        edgecolors="#111827",
        linewidths=1.5,
        ax=ax
    )

    for nid, (x, y) in pos.items():
        ax.text(x, y, labels[nid], fontsize=10, fontweight="bold", ha='center', va='center',
                bbox=label_bboxes[nid], color="#111827")

    nx.draw_networkx_edge_labels(
        display_graph,
        pos,
        edge_labels=edge_labels,
        font_size=7,
        font_color="#1f2937",
        label_pos=0.45,
        ax=ax
    )

    legend_text = (
        "Best path = red | Highly relevant = green | Medium relevance = blue | Low relevance = gray\n"
        "Strong FK edge = dark green thick | Weak FK edge = gray thin | Best path edge = red"
    )
    ax.text(0.01, 0.06, legend_text, transform=ax.transAxes,
            fontsize=8, color="#334155", va='bottom', wrap=True)

    return fig


# -- Build DB dropdown list ------------------------------------

def get_db_list():
    """Get all available database IDs for the dropdown."""
    try:
        p = load_pipeline()
        dbs = sorted(p["schemas"].keys())
        return dbs
    except Exception:
        return ["concert_singer", "department_management"]


# -- Gradio UI ------------------------------------------------

def build_demo():
    db_list = get_db_list()

    # Example queries per database
    examples = [
        ["How many singers do we have?",                          "concert_singer"],
        ["List all singers and the concerts they performed in.",  "concert_singer"],
        ["How many heads of the departments are older than 56?",  "department_management"],
        ["What departments are managed by temporary acting heads?","department_management"],
        ["What are the names of students enrolled in more than 2 courses?",
         "student_transcripts_tracking"],
        ["How many superheroes have a height above average?",     "superhero"],
        ["What is the average age of players in formula 1?",      "formula_1"],
        ["List all card games released after 2010.",              "card_games"],
    ]

    all_known_dbs = sorted(list(set(db_list + [ex[1] for ex in examples])))

    theme = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
    css = """
    .step-box textarea { font-family: monospace !important; font-size: 13px !important; }
    .sql-box textarea { font-family: monospace !important; font-size: 15px !important;
                        background: #1e1e2e !important; color: #cdd6f4 !important; }
    .title-box { text-align: center; padding: 20px 0 10px 0; }
    footer { display: none !important; }
    """

    with gr.Blocks(title="ConSQL -- Constraint-Guided Text-to-SQL") as demo:

        # -- Header -------------------------------------------
        gr.HTML("""
        <div class="title-box">
          <h1 style="font-size:2.2em; color:#1e40af; margin-bottom:4px;">
              ConSQL
          </h1>
          <p style="font-size:1.1em; color:#475569;">
            Constraint-Guided Join Path Selection for Text-to-SQL
          </p>
          <p style="font-size:0.9em; color:#94a3b8;">
            Databases: Spider (166) + BIRD Mini-Dev (11) &nbsp;|&nbsp;
            Model: Flan-T5-small &nbsp;|&nbsp;
            Encoder: SBERT all-MiniLM-L6-v2
          </p>
        </div>
        """)

        # -- Input Section -------------------------------------
        with gr.Row():
            with gr.Column(scale=3):
                question_input = gr.Textbox(
                    label="Natural Language Question",
                    placeholder="e.g. How many singers are older than 25?",
                    lines=2
                )
            with gr.Column(scale=1):
                db_dropdown = gr.Dropdown(
                    choices=all_known_dbs,
                    value=all_known_dbs[0] if all_known_dbs else "concert_singer",
                    label="Select Database",
                    allow_custom_value=True
                )

        with gr.Row():
            run_btn   = gr.Button("  Generate SQL", variant="primary", scale=3)
            clear_btn = gr.Button("   Clear", scale=1)

        # -- Pipeline Steps (tabs -- replaces stacked accordions) -
        gr.Markdown("###   Pipeline Steps")

        with gr.Tabs():
            with gr.Tab("Schema Linking"):
                gr.Markdown("*Identifies relevant tables and columns using semantic similarity*")
                step1_out = gr.Textbox(label="Schema Linking Output",
                                       lines=12, elem_classes=["step-box"])

            with gr.Tab("Graph"):
                gr.Markdown("*Builds a weighted FK graph from the database schema based on the query relevance*")
                schema_graph_plot = gr.Plot(label="Query-Specific Schema Graph",
                                           show_label=True)
                step2_out = gr.Textbox(label="Graph Structure",
                                       lines=10, elem_classes=["step-box"])

            with gr.Tab("Paths"):
                gr.Markdown("*Enumerates all valid join paths via DFS + removes invalid ones*")
                step3_out = gr.Textbox(label="Candidate Paths",
                                       lines=12, elem_classes=["step-box"])

            with gr.Tab("Scoring"):
                gr.Markdown("*Scores all paths: Score =   S_sem +   S_struct     L(p)*")
                step4_out = gr.Textbox(label="Scored Paths (ranked)",
                                       lines=12, elem_classes=["step-box"])

            with gr.Tab("Best Path"):
                gr.Markdown("*The highest-scoring path selected as join strategy*")
                step5_out = gr.Textbox(label="Best Path",
                                       lines=10, elem_classes=["step-box"])

            with gr.Tab("Prompt"):
                gr.Markdown("*Structured prompt combining query + best path + schema context*")
                step6_out = gr.Textbox(label="Serialized Context Prompt",
                                       lines=6, elem_classes=["step-box"])

            with gr.Tab("SQL"):
                gr.Markdown("###   Generated SQL Query")
                step7_out = gr.Textbox(
                    label="Output SQL + Timing",
                    lines=10,
                    elem_classes=["sql-box"]
                )
                sql_query_out = gr.Code(
                    label="SQL Query (raw)",
                    language="sql",
                    lines=10,
                    interactive=False
                )

        # -- Examples -----------------------------------------
        gr.Markdown("###   Example Queries")
        gr.Examples(
            examples=examples,
            inputs=[question_input, db_dropdown],
            label="Click any example to load it"
        )

        # -- Button actions ------------------------------------
        run_btn.click(
            fn=run_pipeline,
            inputs=[question_input, db_dropdown],
            outputs=[step1_out, step2_out, schema_graph_plot, step3_out,
                     step4_out, step5_out, step6_out, step7_out, sql_query_out],
            show_progress=True
        )
        clear_btn.click(
            fn=lambda: ("", "", None, "", "", "", "", "", ""),
            outputs=[step1_out, step2_out, schema_graph_plot, step3_out,
                     step4_out, step5_out, step6_out, step7_out, sql_query_out]
        )

        # -- Footer info ---------------------------------------
        gr.HTML("""
        <div style="text-align:center; padding:20px 0 0 0; color:#94a3b8; font-size:0.85em;">
          Research: Enhancing Text-to-SQL with Constraint-Guided Join Path Selection
        </div>
        """)

    return demo, theme, css


def find_open_port(base_port: int = 7860, max_port: int = 7880) -> int:
    for port in range(base_port, max_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise OSError(f"No open port found in range {base_port}-{max_port}")


def get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  ConSQL Interactive Demo")
    print("="*60)
    print(f"\n  Running from: {Path(__file__).resolve()}")
    print(f"  Current working directory: {Path.cwd()}")
    print("\n  Loading pipeline (first run takes ~30 seconds)...")
    load_pipeline()   # pre-load before UI starts

    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    try:
        server_port = find_open_port(server_port)
    except OSError as exc:
        print(f"\n  ERROR: {exc}")
        raise

    local_ip = get_local_ip()
    print(f"\n  Starting Gradio interface on port {server_port}...")
    print(f"  This PC (Local):     http://localhost:{server_port}")
    print(f"  All Devices (Wi-Fi): http://{local_ip}:{server_port}\n")
    demo, theme, css = build_demo()
    share_public = os.environ.get("GRADIO_SHARE", "false").lower() in ("1", "true", "yes")
    demo.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        theme=theme,
        css=css,
        share=share_public,
        show_error=True
    )