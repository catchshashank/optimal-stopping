const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.title = "Optimal Stopping Replication Study";

// ── Palette ────────────────────────────────────────────────────────────────────
const NAVY      = "1C2B4A";
const TEAL      = "0D9488";
const TEAL_LT   = "5EEAD4";
const OFFWHITE  = "F0F4F8";
const WHITE     = "FFFFFF";
const MID_GRAY  = "64748B";
const LT_GRAY   = "E2E8F0";
const AMBER     = "F59E0B";
const RED_SOFT  = "EF4444";
const GREEN     = "10B981";

const mkShadow = () => ({ type: "outer", blur: 8, offset: 3, angle: 135, color: "000000", opacity: 0.10 });

// ── Reusable helpers ───────────────────────────────────────────────────────────
function darkSlide(pres) {
  const s = pres.addSlide();
  s.background = { color: NAVY };
  return s;
}
function lightSlide(pres) {
  const s = pres.addSlide();
  s.background = { color: OFFWHITE };
  return s;
}
function addLeftAccent(slide, color) {
  slide.addShape(slide.pres ? slide.pres.shapes.RECTANGLE : pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.10, h: 5.625,
    fill: { color: color || TEAL }, line: { color: color || TEAL }
  });
}
function slideLabel(slide, text, dark) {
  slide.addText(text.toUpperCase(), {
    x: 0.35, y: 0.22, w: 9.3, h: 0.28,
    fontSize: 9, color: dark ? TEAL_LT : TEAL, fontFace: "Calibri",
    charSpacing: 3, bold: true, margin: 0
  });
}
function slideTitle(slide, text, dark) {
  slide.addText(text, {
    x: 0.35, y: 0.55, w: 9.3, h: 0.75,
    fontSize: 28, bold: true, fontFace: "Calibri",
    color: dark ? WHITE : NAVY, valign: "top", margin: 0
  });
}
function dividerLine(slide, y, dark) {
  slide.addShape(pres.shapes.LINE, {
    x: 0.35, y: y, w: 9.3, h: 0,
    line: { color: dark ? "2E4A6E" : LT_GRAY, width: 1 }
  });
}

// ──────────────────────────────────────────────────────────────────────────────
// SLIDE 1 — TITLE
// ──────────────────────────────────────────────────────────────────────────────
{
  const s = darkSlide(pres);
  // Left accent bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.10, h: 5.625,
    fill: { color: TEAL }, line: { color: TEAL }
  });
  // Teal decorative block top-right
  s.addShape(pres.shapes.RECTANGLE, {
    x: 8.8, y: 0, w: 1.2, h: 1.1,
    fill: { color: TEAL, transparency: 75 }, line: { color: TEAL, transparency: 75 }
  });

  s.addText("When Should an Agent Stop Listening?", {
    x: 0.35, y: 1.05, w: 9.3, h: 1.5,
    fontSize: 38, bold: true, color: WHITE, fontFace: "Calibri", valign: "middle", margin: 0
  });
  s.addText("Replicating and Stress-Testing LLM-Based Optimal Stopping\nAcross Conversational Domains", {
    x: 0.35, y: 2.65, w: 9.3, h: 0.9,
    fontSize: 17, color: TEAL_LT, fontFace: "Calibri", valign: "top", margin: 0
  });
  dividerLine(s, 3.7, true);
  s.addText("MIT PhD Seminar  ·  May 2026", {
    x: 0.35, y: 3.85, w: 9.3, h: 0.35,
    fontSize: 12, color: "8BA3C0", fontFace: "Calibri", margin: 0
  });
  s.addText("Based on Manzoor, Ascarza & Netzer (2025)", {
    x: 0.35, y: 4.25, w: 9.3, h: 0.3,
    fontSize: 11, color: "5E7A99", fontFace: "Calibri", italic: true, margin: 0
  });
}

// ──────────────────────────────────────────────────────────────────────────────
// SLIDE 2 — THE PROBLEM
// ──────────────────────────────────────────────────────────────────────────────
{
  const s = lightSlide(pres);
  addLeftAccent(s);
  slideLabel(s, "Act 1 · The Framework");
  slideTitle(s, "A Sales Call Is a Sequential Decision Problem");
  dividerLine(s, 1.38);

  // Left column — text
  s.addText("Every live sales call poses a real-time trade-off:", {
    x: 0.35, y: 1.5, w: 4.5, h: 0.4,
    fontSize: 13, color: NAVY, fontFace: "Calibri", bold: true, margin: 0
  });
  const leftBullets = [
    "Continuing costs time (agent hours, infrastructure)",
    "Ending early risks losing a sale",
    "The optimal stopping point is unknown in advance",
  ];
  leftBullets.forEach((txt, i) => {
    s.addText([{ text: txt, options: { bullet: true } }], {
      x: 0.45, y: 1.95 + i * 0.45, w: 4.3, h: 0.4,
      fontSize: 13, color: NAVY, fontFace: "Calibri", margin: 0
    });
  });

  s.addText("The Bermudan option framing:", {
    x: 0.35, y: 3.35, w: 4.5, h: 0.35,
    fontSize: 12, color: MID_GRAY, fontFace: "Calibri", bold: true, italic: true, margin: 0
  });
  s.addText("At each checkpoint m, the agent decides:\nQuit now (take certain reward) or continue (wait for better information)?", {
    x: 0.35, y: 3.72, w: 4.5, h: 0.75,
    fontSize: 12, color: MID_GRAY, fontFace: "Calibri", italic: true, margin: 0
  });

  // Right column — timeline diagram
  const timelineY = 1.6;
  const timelineX = 5.2;
  const timelineW = 4.3;

  // Timeline bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: timelineX, y: timelineY + 0.55, w: timelineW, h: 0.10,
    fill: { color: LT_GRAY }, line: { color: LT_GRAY }
  });

  // Checkpoints
  const checkpoints = [
    { frac: 0,    label: "Start",  sub: "0s" },
    { frac: 0.33, label: "m₁",    sub: "~33%" },
    { frac: 0.65, label: "m₂",    sub: "~65%" },
    { frac: 1.0,  label: "End",   sub: "100%" },
  ];
  const cpColors = [LT_GRAY, TEAL, TEAL, NAVY];
  checkpoints.forEach((cp, i) => {
    const cx = timelineX + cp.frac * timelineW;
    s.addShape(pres.shapes.OVAL, {
      x: cx - 0.13, y: timelineY + 0.4, w: 0.26, h: 0.26,
      fill: { color: cpColors[i] }, line: { color: cpColors[i] }
    });
    s.addText(cp.label, {
      x: cx - 0.4, y: timelineY + 0.75, w: 0.8, h: 0.3,
      fontSize: 11, bold: true, color: NAVY, fontFace: "Calibri", align: "center", margin: 0
    });
    s.addText(cp.sub, {
      x: cx - 0.4, y: timelineY + 1.05, w: 0.8, h: 0.25,
      fontSize: 9, color: MID_GRAY, fontFace: "Calibri", align: "center", margin: 0
    });
  });

  // Decision boxes
  const decBoxes = [
    { x: timelineX + 0.33 * timelineW - 0.1, y: timelineY - 0.65, label: "Quit or continue?", color: TEAL },
    { x: timelineX + 0.65 * timelineW - 0.1, y: timelineY - 0.65, label: "Quit or continue?", color: TEAL },
  ];
  decBoxes.forEach(b => {
    s.addShape(pres.shapes.RECTANGLE, {
      x: b.x - 0.55, y: b.y, w: 1.3, h: 0.42,
      fill: { color: TEAL, transparency: 20 }, line: { color: TEAL }
    });
    s.addText(b.label, {
      x: b.x - 0.55, y: b.y, w: 1.3, h: 0.42,
      fontSize: 9, color: WHITE, fontFace: "Calibri", align: "center", valign: "middle", bold: true, margin: 0
    });
  });

  // Outcome box
  s.addShape(pres.shapes.RECTANGLE, {
    x: timelineX + timelineW - 0.2, y: timelineY - 0.65, w: 1.0, h: 0.42,
    fill: { color: NAVY, transparency: 10 }, line: { color: NAVY }
  });
  s.addText("Outcome\nRevealed", {
    x: timelineX + timelineW - 0.2, y: timelineY - 0.65, w: 1.0, h: 0.42,
    fontSize: 8, color: WHITE, fontFace: "Calibri", align: "center", valign: "middle", bold: true, margin: 0
  });

  // Reward formula box
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: 3.9, w: 4.3, h: 1.3,
    fill: { color: NAVY, transparency: 5 }, line: { color: NAVY }, shadow: mkShadow()
  });
  s.addText("Reward  =  sales × benefit  −  total_time × cost", {
    x: 5.2, y: 4.05, w: 4.3, h: 0.45,
    fontSize: 14, bold: true, color: TEAL_LT, fontFace: "Calibri", align: "center", margin: 0
  });
  s.addText("Threshold policy: quit if  P(sale | transcript)  <  θ", {
    x: 5.2, y: 4.58, w: 4.3, h: 0.45,
    fontSize: 12, color: "8BA3C0", fontFace: "Calibri", align: "center", italic: true, margin: 0
  });
}

// ──────────────────────────────────────────────────────────────────────────────
// SLIDE 3 — THE PIPELINE
// ──────────────────────────────────────────────────────────────────────────────
{
  const s = lightSlide(pres);
  addLeftAccent(s);
  slideLabel(s, "Act 1 · The Framework");
  slideTitle(s, "Four-Component Pipeline");
  dividerLine(s, 1.38);

  const steps = [
    { icon: "①", title: "Partial Transcript", body: "Conversation up to window m\nformatted as prompt", color: TEAL },
    { icon: "②", title: "GPT-4o (Zero-Shot)", body: "\"Will this call end in a sale?\"\nlogprobs → P(yes) normalized", color: TEAL },
    { icon: "③", title: "Backward Induction", body: "Tune θ_m₂ first, then θ_m₁\nGrid search on validation set", color: NAVY },
    { icon: "④", title: "Evaluate on Test", body: "Reward vs. baseline\nSales retained · Time saved", color: NAVY },
  ];

  const bw = 2.0, bh = 2.8, gap = 0.15;
  const startX = 0.5;

  steps.forEach((st, i) => {
    const bx = startX + i * (bw + gap);
    const by = 1.55;

    s.addShape(pres.shapes.RECTANGLE, {
      x: bx, y: by, w: bw, h: bh,
      fill: { color: WHITE }, line: { color: LT_GRAY }, shadow: mkShadow()
    });
    // Color top bar
    s.addShape(pres.shapes.RECTANGLE, {
      x: bx, y: by, w: bw, h: 0.45,
      fill: { color: st.color }, line: { color: st.color }
    });
    s.addText(st.icon + "  " + st.title, {
      x: bx + 0.1, y: by, w: bw - 0.2, h: 0.45,
      fontSize: 12, bold: true, color: WHITE, fontFace: "Calibri", valign: "middle", margin: 0
    });
    s.addText(st.body, {
      x: bx + 0.15, y: by + 0.55, w: bw - 0.3, h: bh - 0.7,
      fontSize: 12, color: NAVY, fontFace: "Calibri", valign: "top", margin: 0
    });

    // Arrow between cards
    if (i < steps.length - 1) {
      s.addShape(pres.shapes.LINE, {
        x: bx + bw + 0.02, y: by + bh / 2,
        w: gap - 0.04, h: 0,
        line: { color: MID_GRAY, width: 1.5 }
      });
    }
  });

  s.addText("θ tuned via backward induction: m₂ threshold first, then m₁ conditional on θ_m₂  ·  10,000-point grid search", {
    x: 0.35, y: 4.7, w: 9.3, h: 0.3,
    fontSize: 10, color: MID_GRAY, italic: true, fontFace: "Calibri", align: "center", margin: 0
  });
}

// ──────────────────────────────────────────────────────────────────────────────
// SLIDE 4 — ORIGINAL STUDY RESULTS
// ──────────────────────────────────────────────────────────────────────────────
{
  const s = lightSlide(pres);
  addLeftAccent(s);
  slideLabel(s, "Act 1 · The Framework");
  slideTitle(s, "What Manzoor et al. (2025) Found");
  dividerLine(s, 1.38);

  // Highlight box
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.35, y: 1.5, w: 9.3, h: 0.75,
    fill: { color: TEAL, transparency: 85 }, line: { color: TEAL }
  });
  s.addText("Best configuration: 90s / 180s windows · cost = 0.01 · benefit = 10 · GPT-4o (zero-shot)", {
    x: 0.45, y: 1.5, w: 9.1, h: 0.75,
    fontSize: 13, color: NAVY, fontFace: "Calibri", bold: true, valign: "middle", margin: 0
  });

  // Table
  const rows = [
    [
      { text: "Run", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
      { text: "Windows", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
      { text: "AUC m₁", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
      { text: "AUC m₂", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
      { text: "Sales retained", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
      { text: "Time saved", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
      { text: "Reward gained", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
    ],
    ["Original", "45s / 60s", "0.520", "0.488", "0 / 27", "88.6%", "+1,328"],
    ["RF1 × DP1", "115s / 230s", "0.585", "0.447", "24 / 27", "5.0%", "−21"],
    ["RF2 × DP1", "115s / 230s", "0.585", "0.447", "0 / 27", "71.3%", "+373"],
    [
      { text: "RF1 × DP2 ★", options: { bold: true, fill: { color: "D1FAE5" }, color: "065F46" } },
      { text: "90s / 180s", options:   { bold: true, fill: { color: "D1FAE5" }, color: "065F46" } },
      { text: "0.753", options:        { bold: true, fill: { color: "D1FAE5" }, color: "065F46" } },
      { text: "0.532", options:        { bold: true, fill: { color: "D1FAE5" }, color: "065F46" } },
      { text: "27 / 27", options:      { bold: true, fill: { color: "D1FAE5" }, color: "065F46" } },
      { text: "1.8%", options:         { bold: true, fill: { color: "D1FAE5" }, color: "065F46" } },
      { text: "+3", options:           { bold: true, fill: { color: "D1FAE5" }, color: "065F46" } },
    ],
    ["RF2 × DP2", "90s / 180s", "0.753", "0.532", "0 / 27", "77.5%", "+429"],
  ];

  s.addTable(rows, {
    x: 0.35, y: 2.35, w: 9.3, h: 2.7,
    fontSize: 11, fontFace: "Calibri", color: NAVY,
    border: { pt: 0.5, color: LT_GRAY },
    rowH: 0.42,
    colW: [1.4, 1.3, 0.95, 0.95, 1.3, 1.2, 1.4],
    align: "center",
  });
  s.addText("★ Only run retaining all sales while improving reward  ·  NL Negotiation dataset (178 conversations, GPT-4o zero-shot)", {
    x: 0.35, y: 5.15, w: 9.3, h: 0.3,
    fontSize: 9, color: MID_GRAY, italic: true, fontFace: "Calibri", margin: 0
  });
}

// ──────────────────────────────────────────────────────────────────────────────
// SLIDE 5 — GENERALIZATION QUESTION (dark divider)
// ──────────────────────────────────────────────────────────────────────────────
{
  const s = darkSlide(pres);
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.10, h: 5.625,
    fill: { color: TEAL }, line: { color: TEAL }
  });
  s.addText("The Generalization Question", {
    x: 0.35, y: 1.1, w: 9.3, h: 0.6,
    fontSize: 14, color: TEAL_LT, fontFace: "Calibri", bold: true, charSpacing: 2, margin: 0
  });
  s.addText("Is the 90s / AUC = 0.75 result a property\nof this dataset, or of conversational structure generally?", {
    x: 0.35, y: 1.8, w: 9.3, h: 1.4,
    fontSize: 30, bold: true, color: WHITE, fontFace: "Calibri", valign: "top", margin: 0
  });
  dividerLine(s, 3.4, true);
  s.addText("We stress-test on a qualitatively different domain: real car dealership sales calls with diarized timestamps.", {
    x: 0.35, y: 3.6, w: 9.3, h: 0.55,
    fontSize: 15, color: "8BA3C0", fontFace: "Calibri", italic: true, margin: 0
  });
}

// ──────────────────────────────────────────────────────────────────────────────
// SLIDE 6 — NL REPLICATION SUMMARY
// ──────────────────────────────────────────────────────────────────────────────
{
  const s = lightSlide(pres);
  addLeftAccent(s);
  slideLabel(s, "Act 2 · NL Replication");
  slideTitle(s, "NL Dataset Confirms the Original");
  dividerLine(s, 1.38);

  // Three stat cards
  const stats = [
    { val: "178", label: "Conversations", sub: "105 sale · 73 no-sale" },
    { val: "389s", label: "Avg Duration", sub: "Range: 46s – 946s" },
    { val: "0.753", label: "Best AUC (90s)", sub: "3-in-4 pairs correctly ranked" },
  ];
  stats.forEach((st, i) => {
    const bx = 0.35 + i * 3.2;
    s.addShape(pres.shapes.RECTANGLE, {
      x: bx, y: 1.5, w: 2.95, h: 1.5,
      fill: { color: WHITE }, line: { color: LT_GRAY }, shadow: mkShadow()
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: bx, y: 1.5, w: 2.95, h: 0.10,
      fill: { color: TEAL }, line: { color: TEAL }
    });
    s.addText(st.val, {
      x: bx, y: 1.65, w: 2.95, h: 0.75,
      fontSize: 38, bold: true, color: NAVY, fontFace: "Calibri", align: "center", valign: "middle", margin: 0
    });
    s.addText(st.label, {
      x: bx, y: 2.45, w: 2.95, h: 0.3,
      fontSize: 12, bold: true, color: NAVY, fontFace: "Calibri", align: "center", margin: 0
    });
    s.addText(st.sub, {
      x: bx, y: 2.77, w: 2.95, h: 0.22,
      fontSize: 10, color: MID_GRAY, fontFace: "Calibri", align: "center", margin: 0
    });
  });

  // Key finding box
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.35, y: 3.2, w: 9.3, h: 1.75,
    fill: { color: NAVY }, line: { color: NAVY }, shadow: mkShadow()
  });
  s.addText("Key empirical regularity from both studies:", {
    x: 0.55, y: 3.32, w: 9.0, h: 0.35,
    fontSize: 12, color: TEAL_LT, bold: true, fontFace: "Calibri", margin: 0
  });
  const findings = [
    "AUC < 0.65 at m₁  →  agent collapses: quits everything or nothing useful",
    "AUC ≥ 0.65  →  meaningful separation; threshold tuning can retain sales",
    "90s = the earliest window where GPT-4o crosses from noise into genuine signal for NL data",
  ];
  findings.forEach((f, i) => {
    s.addText([{ text: f, options: { bullet: true } }], {
      x: 0.55, y: 3.72 + i * 0.36, w: 9.0, h: 0.34,
      fontSize: 12, color: WHITE, fontFace: "Calibri", margin: 0
    });
  });
}

// ──────────────────────────────────────────────────────────────────────────────
// SLIDE 7 — DEALERSHIPS DATASET
// ──────────────────────────────────────────────────────────────────────────────
{
  const s = lightSlide(pres);
  addLeftAccent(s);
  slideLabel(s, "Act 3 · Dealerships as Stress Test");
  slideTitle(s, "A Structurally Different Dataset");
  dividerLine(s, 1.38);

  // Comparison table
  const rows = [
    [
      { text: "", options: { fill: { color: OFFWHITE } } },
      { text: "NL Negotiations", options: { bold: true, fill: { color: NAVY }, color: WHITE, align: "center" } },
      { text: "Dealerships", options: { bold: true, fill: { color: TEAL }, color: WHITE, align: "center" } },
    ],
    ["N conversations",    "178",       "48"],
    ["Avg duration",       "389s",      "497s  (std 356s)"],
    ["Timestamps",         "Estimated (proportional)", "Real diarized"],
    ["No-sale median",     "~280s",     "163s"],
    ["Sale median",        "~400s",     "716s"],
    ["Duration as predictor", "Weak",  { text: "Strong ⚠️", options: { bold: true, color: RED_SOFT } }],
    ["Test set size",      "45 calls",  { text: "12 calls", options: { bold: true, color: AMBER } }],
  ];

  s.addTable(rows, {
    x: 0.35, y: 1.5, w: 9.3, h: 3.7,
    fontSize: 12, fontFace: "Calibri", color: NAVY,
    border: { pt: 0.5, color: LT_GRAY },
    rowH: 0.44,
    colW: [3.5, 2.9, 2.9],
    align: "left",
  });

  s.addText("With only 12 test conversations, one misclassification shifts 'sales lost' by 17% — results are directional, not definitive.", {
    x: 0.35, y: 5.3, w: 9.3, h: 0.25,
    fontSize: 9, color: AMBER, italic: true, fontFace: "Calibri", margin: 0
  });
}

// ──────────────────────────────────────────────────────────────────────────────
// SLIDE 8 — BIMODAL DISTRIBUTION
// ──────────────────────────────────────────────────────────────────────────────
{
  const s = lightSlide(pres);
  addLeftAccent(s);
  slideLabel(s, "Act 3 · Dealerships as Stress Test");
  slideTitle(s, "The Bimodal Duration Problem");
  dividerLine(s, 1.38);

  // Bar chart — call duration distribution
  const chartData = [{
    name: "No-sale",
    labels: ["<100s", "100–175s", "175–260s", "260–350s", "350–400s", "400–600s", "600–700s", ">700s"],
    values: [2, 13, 1, 1, 1, 1, 2, 4],
  }, {
    name: "Sale",
    labels: ["<100s", "100–175s", "175–260s", "260–350s", "350–400s", "400–600s", "600–700s", ">700s"],
    values: [0, 2, 1, 1, 1, 2, 3, 10],
  }];

  s.addChart(pres.charts.BAR, chartData, {
    x: 0.35, y: 1.5, w: 5.8, h: 3.5,
    barDir: "col", barGrouping: "stacked",
    chartColors: [RED_SOFT, TEAL],
    chartArea: { fill: { color: OFFWHITE } },
    catAxisLabelColor: MID_GRAY,
    valAxisLabelColor: MID_GRAY,
    valGridLine: { color: LT_GRAY, size: 0.5 },
    catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendFontSize: 10,
    showTitle: false,
    dataLabelFontSize: 9,
  });

  // Right annotations
  const annotations = [
    { y: 1.55, color: RED_SOFT,  bold: true,  txt: "Short calls = no-sales" },
    { y: 1.95, color: MID_GRAY,  bold: false, txt: "No-sale median: 163s" },
    { y: 2.45, color: TEAL,      bold: true,  txt: "Long calls = sales" },
    { y: 2.85, color: MID_GRAY,  bold: false, txt: "Sale median: 716s" },
    { y: 3.35, color: AMBER,     bold: true,  txt: "175–400s window range:" },
    { y: 3.75, color: AMBER,     bold: false, txt: "only 6 calls total ⚠️" },
    { y: 4.2,  color: NAVY,      bold: true,  txt: "47.9% of calls end before" },
    { y: 4.6,  color: NAVY,      bold: false, txt: "any 400s window is reached" },
  ];
  annotations.forEach(a => {
    s.addText(a.txt, {
      x: 6.35, y: a.y, w: 3.3, h: 0.35,
      fontSize: 12, color: a.color, bold: a.bold, fontFace: "Calibri", valign: "middle", margin: 0
    });
  });
}

// ──────────────────────────────────────────────────────────────────────────────
// SLIDE 9 — EXPERIMENT PROGRESSION
// ──────────────────────────────────────────────────────────────────────────────
{
  const s = lightSlide(pres);
  addLeftAccent(s);
  slideLabel(s, "Act 3 · Dealerships as Stress Test");
  slideTitle(s, "Three Experiments, One Insight");
  dividerLine(s, 1.38);

  const exps = [
    {
      num: "01",
      title: "Early Windows (2×2 Grid)",
      windows: "90s / 180s and 115s / 230s",
      finding: "AUC < 0.5 at 90s. Windows too early — only greeting and inventory-check language present. Agent fails.",
      verdict: "✗ Too early",
      vcolor: RED_SOFT,
    },
    {
      num: "02",
      title: "4-Window Sequential",
      windows: "175s, 260s, 350s, 400s",
      finding: "AUC 0.64–0.67 at 175s. Improves but 350s collapses to 0.333. Backward induction over 4 windows.",
      verdict: "~ Partial signal",
      vcolor: AMBER,
    },
    {
      num: "03",
      title: "3-Pair Grid (Main)",
      windows: "WP1(175/260), WP2(260/350), WP3(350/400) × 4 reward functions",
      finding: "Conditional AUC reveals 0.833 at 175s. WP1×RF3 retains all sales, saves 3.1% time.",
      verdict: "✓ Clear pattern",
      vcolor: GREEN,
    },
  ];

  exps.forEach((e, i) => {
    const by = 1.55 + i * 1.3;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.35, y: by, w: 9.3, h: 1.15,
      fill: { color: WHITE }, line: { color: LT_GRAY }, shadow: mkShadow()
    });
    // Number badge
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.35, y: by, w: 0.72, h: 1.15,
      fill: { color: NAVY }, line: { color: NAVY }
    });
    s.addText(e.num, {
      x: 0.35, y: by, w: 0.72, h: 1.15,
      fontSize: 22, bold: true, color: TEAL_LT, fontFace: "Calibri", align: "center", valign: "middle", margin: 0
    });
    s.addText(e.title, {
      x: 1.15, y: by + 0.08, w: 5.5, h: 0.35,
      fontSize: 13, bold: true, color: NAVY, fontFace: "Calibri", margin: 0
    });
    s.addText("Windows: " + e.windows, {
      x: 1.15, y: by + 0.4, w: 5.5, h: 0.28,
      fontSize: 10, color: MID_GRAY, italic: true, fontFace: "Calibri", margin: 0
    });
    s.addText(e.finding, {
      x: 1.15, y: by + 0.68, w: 5.5, h: 0.38,
      fontSize: 10, color: NAVY, fontFace: "Calibri", margin: 0
    });
    // Verdict pill
    s.addShape(pres.shapes.RECTANGLE, {
      x: 6.85, y: by + 0.35, w: 2.6, h: 0.45,
      fill: { color: e.vcolor, transparency: 80 }, line: { color: e.vcolor }
    });
    s.addText(e.verdict, {
      x: 6.85, y: by + 0.35, w: 2.6, h: 0.45,
      fontSize: 11, bold: true, color: e.vcolor, fontFace: "Calibri", align: "center", valign: "middle", margin: 0
    });
  });
}

// ──────────────────────────────────────────────────────────────────────────────
// SLIDE 10 — THE CONDITIONAL AUC FIX
// ──────────────────────────────────────────────────────────────────────────────
{
  const s = lightSlide(pres);
  addLeftAccent(s);
  slideLabel(s, "Act 3 · Methodology");
  slideTitle(s, "Conditional AUC: The Right Metric");
  dividerLine(s, 1.38);

  // Problem box
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.35, y: 1.5, w: 4.45, h: 1.7,
    fill: { color: "FEF2F2" }, line: { color: RED_SOFT }, shadow: mkShadow()
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.35, y: 1.5, w: 4.45, h: 0.38,
    fill: { color: RED_SOFT }, line: { color: RED_SOFT }
  });
  s.addText("❌  Unconditional AUC (before)", {
    x: 0.45, y: 1.5, w: 4.25, h: 0.38,
    fontSize: 12, bold: true, color: WHITE, fontFace: "Calibri", valign: "middle", margin: 0
  });
  s.addText("Calls ending before window m get\nprob_yes = 0 via fillna(0).\n\nThis conflates two populations: calls\nstill live at m and calls already done.\nAUC is suppressed or inflated artificially.", {
    x: 0.45, y: 1.95, w: 4.15, h: 1.2,
    fontSize: 11, color: NAVY, fontFace: "Calibri", margin: 0
  });

  // Fix box
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: 1.5, w: 4.45, h: 1.7,
    fill: { color: "F0FDF4" }, line: { color: GREEN }, shadow: mkShadow()
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: 1.5, w: 4.45, h: 0.38,
    fill: { color: GREEN }, line: { color: GREEN }
  });
  s.addText("✓  Conditional AUC (after)", {
    x: 5.3, y: 1.5, w: 4.25, h: 0.38,
    fontSize: 12, bold: true, color: WHITE, fontFace: "Calibri", valign: "middle", margin: 0
  });
  s.addText("Compute AUC only on conversations\nwhere duration ≥ m.\n\nMeasures GPT-4o's true discriminability\namong calls that are actually live\nat the decision point.", {
    x: 5.3, y: 1.95, w: 4.15, h: 1.2,
    fontSize: 11, color: NAVY, fontFace: "Calibri", margin: 0
  });

  // Arrow
  s.addShape(pres.shapes.LINE, {
    x: 4.8, y: 2.35, w: 0.4, h: 0,
    line: { color: MID_GRAY, width: 2 }
  });

  // Before/After table
  const aucs = [
    ["Window", "Unconditional", "Conditional", "n (live)"],
    ["175s", "0.611", { text: "0.833  ↑", options: { bold: true, color: GREEN } }, "10 / 12"],
    ["260s", "0.444", { text: "0.533  ↑", options: { bold: true, color: GREEN } }, "8 / 12"],
    ["350s", "0.361", { text: "0.400  ↑", options: { color: AMBER } }, "8 / 12"],
    ["400s", "0.583", { text: "0.900  ↑", options: { bold: true, color: GREEN } }, "7 / 12"],
  ];
  s.addTable(aucs, {
    x: 0.35, y: 3.3, w: 9.3, h: 2.0,
    fontSize: 12, fontFace: "Calibri", color: NAVY,
    border: { pt: 0.5, color: LT_GRAY },
    rowH: 0.38,
    colW: [1.8, 2.5, 2.9, 2.1],
    align: "center",
  });

  s.addText("This applies generally to any survival-indexed classification task — a broader methodological contribution.", {
    x: 0.35, y: 5.3, w: 9.3, h: 0.25,
    fontSize: 9, color: MID_GRAY, italic: true, fontFace: "Calibri", margin: 0
  });
}

// ──────────────────────────────────────────────────────────────────────────────
// SLIDE 11 — RESULTS 3-PAIR GRID
// ──────────────────────────────────────────────────────────────────────────────
{
  const s = lightSlide(pres);
  addLeftAccent(s);
  slideLabel(s, "Act 3 · Results");
  slideTitle(s, "12-Run Grid Results (Conditional AUC)");
  dividerLine(s, 1.38);

  const rows = [
    [
      { text: "Window Pair", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
      { text: "Reward Fn", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
      { text: "AUC m₁", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
      { text: "AUC m₂", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
      { text: "Sales lost", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
      { text: "Time saved", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
      { text: "Reward Δ", options: { bold: true, fill: { color: NAVY }, color: WHITE } },
    ],
    // WP1
    [
      { text: "WP1  175/260s", options: { fill: { color: "EFF6FF" } } },
      "RF1 c=0.1 b=10", "0.833", "0.533", { text: "4 / 6", options: { color: RED_SOFT } }, "54.3%", "+378"
    ],
    ["WP1  175/260s", "RF2 c=0.5 b=10", "0.833", "0.533", { text: "5 / 6", options: { color: RED_SOFT } }, "71.9%", "+2,718"],
    [
      { text: "WP1  175/260s", options: { fill: { color: "D1FAE5" } } },
      { text: "RF3 c=1.0 b=1000 ★", options: { bold: true, fill: { color: "D1FAE5" }, color: "065F46" } },
      { text: "0.833", options: { bold: true, fill: { color: "D1FAE5" }, color: "065F46" } },
      { text: "0.533", options: { bold: true, fill: { color: "D1FAE5" }, color: "065F46" } },
      { text: "0 / 6", options: { bold: true, fill: { color: "D1FAE5" }, color: "065F46" } },
      { text: "3.1%", options: { bold: true, fill: { color: "D1FAE5" }, color: "065F46" } },
      { text: "+235", options: { bold: true, fill: { color: "D1FAE5" }, color: "065F46" } },
    ],
    ["WP1  175/260s", "RF4 c=5.0 b=1000", "0.833", "0.533", { text: "4 / 6", options: { color: RED_SOFT } }, "54.3%", "+16,891†"],
    // WP2
    ["WP2  260/350s", "RF3 c=1.0 b=1000", "0.600", "0.533", "0 / 6", { text: "0.0%", options: { color: AMBER } }, "0"],
    // WP3
    [
      { text: "WP3  350/400s", options: { fill: { color: "FFF7ED" } } },
      { text: "RF3 c=1.0 b=1000", options: { fill: { color: "FFF7ED" } } },
      { text: "0.400 ↓", options: { color: RED_SOFT, fill: { color: "FFF7ED" } } },
      { text: "0.900 ↑", options: { bold: true, color: GREEN, fill: { color: "FFF7ED" } } },
      { text: "0 / 6", options: { fill: { color: "FFF7ED" } } },
      { text: "0.0%", options: { color: AMBER, fill: { color: "FFF7ED" } } },
      { text: "0", options: { fill: { color: "FFF7ED" } } },
    ],
  ];

  s.addTable(rows, {
    x: 0.35, y: 1.5, w: 9.3, h: 3.5,
    fontSize: 10, fontFace: "Calibri", color: NAVY,
    border: { pt: 0.5, color: LT_GRAY },
    rowH: 0.42,
    colW: [1.6, 1.9, 0.85, 0.85, 0.85, 1.1, 1.15],
    align: "center",
  });
  s.addText("★ Only config retaining all sales while improving reward  ·  † RF4 reward gains are time-driven, not sales-driven — 4 of 6 sales dropped", {
    x: 0.35, y: 5.1, w: 9.3, h: 0.3,
    fontSize: 9, color: MID_GRAY, italic: true, fontFace: "Calibri", margin: 0
  });
}

// ──────────────────────────────────────────────────────────────────────────────
// SLIDE 12 — THE 350s DEAD ZONE
// ──────────────────────────────────────────────────────────────────────────────
{
  const s = lightSlide(pres);
  addLeftAccent(s);
  slideLabel(s, "Act 3 · Results");
  slideTitle(s, "The 350s Dead Zone");
  dividerLine(s, 1.38);

  // AUC line chart across windows
  s.addChart(pres.charts.LINE, [
    {
      name: "Unconditional AUC",
      labels: ["175s", "260s", "350s", "400s"],
      values: [0.611, 0.444, 0.361, 0.583],
    },
    {
      name: "Conditional AUC",
      labels: ["175s", "260s", "350s", "400s"],
      values: [0.833, 0.533, 0.400, 0.900],
    },
  ], {
    x: 0.35, y: 1.5, w: 5.5, h: 3.5,
    chartColors: [MID_GRAY, TEAL],
    chartArea: { fill: { color: OFFWHITE } },
    catAxisLabelColor: MID_GRAY,
    valAxisLabelColor: MID_GRAY,
    valAxisMinVal: 0.3, valAxisMaxVal: 1.0,
    valGridLine: { color: LT_GRAY, size: 0.5 },
    catGridLine: { style: "none" },
    lineSize: 2.5,
    showLegend: true, legendPos: "b", legendFontSize: 10,
    showValue: true, dataLabelFontSize: 9, dataLabelColor: NAVY,
    showTitle: false,
  });

  // Annotations right panel
  const notes = [
    { y: 1.55, icon: "↑", color: TEAL,     title: "175s: AUC 0.833",   body: "Opening + early offer language\ncaptured. Strong signal." },
    { y: 2.4,  icon: "↓", color: AMBER,    title: "350s: AUC 0.400",   body: "Below chance. Price anchoring and\nobjections appear in both outcomes." },
    { y: 3.25, icon: "↑", color: GREEN,    title: "400s: AUC 0.900",   body: "Among 7 surviving calls, near-perfect\ndiscrimination. Closing cues very clear." },
    { y: 4.1,  icon: "!", color: RED_SOFT, title: "Structural gap",     body: "Only 6 calls in the 175–400s range.\nThreshold tuning on 9-call val set." },
  ];
  notes.forEach(n => {
    s.addShape(pres.shapes.RECTANGLE, {
      x: 6.05, y: n.y, w: 3.6, h: 0.75,
      fill: { color: WHITE }, line: { color: LT_GRAY }, shadow: mkShadow()
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 6.05, y: n.y, w: 0.12, h: 0.75,
      fill: { color: n.color }, line: { color: n.color }
    });
    s.addText(n.title, {
      x: 6.25, y: n.y + 0.06, w: 3.3, h: 0.28,
      fontSize: 11, bold: true, color: NAVY, fontFace: "Calibri", margin: 0
    });
    s.addText(n.body, {
      x: 6.25, y: n.y + 0.35, w: 3.3, h: 0.35,
      fontSize: 9.5, color: MID_GRAY, fontFace: "Calibri", margin: 0
    });
  });
}

// ──────────────────────────────────────────────────────────────────────────────
// SLIDE 13 — WHAT GENERALIZES?
// ──────────────────────────────────────────────────────────────────────────────
{
  const s = lightSlide(pres);
  addLeftAccent(s);
  slideLabel(s, "Act 4 · Synthesis");
  slideTitle(s, "What Generalizes Across Domains?");
  dividerLine(s, 1.38);

  const items = [
    {
      label: "✓  Generalizes",
      color: GREEN,
      bcolor: "F0FDF4",
      rows: [
        ["Backward induction framework", "Exact optimal policy given known reward and finite horizon"],
        ["AUC > ~0.65 threshold", "Below this, no cost/benefit tuning rescues the agent"],
        ["Logprobs as probability proxy", "Normalized P(yes) is a reliable soft classifier signal"],
      ]
    },
    {
      label: "✗  Domain-specific",
      color: RED_SOFT,
      bcolor: "FEF2F2",
      rows: [
        ["Window placement", "Must reflect call structure — 90s NL ≠ 175s dealerships"],
        ["Cost/benefit calibration", "Ratio drives policy more than model quality"],
        ["AUC magnitude", "Depends on domain, dataset size, and linguistic richness"],
      ]
    },
  ];

  items.forEach((item, col) => {
    const bx = 0.35 + col * 4.85;
    s.addShape(pres.shapes.RECTANGLE, {
      x: bx, y: 1.5, w: 4.6, h: 3.8,
      fill: { color: item.bcolor }, line: { color: item.color }, shadow: mkShadow()
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: bx, y: 1.5, w: 4.6, h: 0.4,
      fill: { color: item.color }, line: { color: item.color }
    });
    s.addText(item.label, {
      x: bx + 0.1, y: 1.5, w: 4.4, h: 0.4,
      fontSize: 13, bold: true, color: WHITE, fontFace: "Calibri", valign: "middle", margin: 0
    });
    item.rows.forEach((r, ri) => {
      s.addText(r[0], {
        x: bx + 0.15, y: 2.0 + ri * 0.9, w: 4.3, h: 0.32,
        fontSize: 12, bold: true, color: NAVY, fontFace: "Calibri", margin: 0
      });
      s.addText(r[1], {
        x: bx + 0.15, y: 2.32 + ri * 0.9, w: 4.3, h: 0.48,
        fontSize: 10.5, color: MID_GRAY, fontFace: "Calibri", margin: 0
      });
    });
  });
}

// ──────────────────────────────────────────────────────────────────────────────
// SLIDE 14 — THE DURATION CONFOUND
// ──────────────────────────────────────────────────────────────────────────────
{
  const s = lightSlide(pres);
  addLeftAccent(s);
  slideLabel(s, "Act 4 · Open Questions");
  slideTitle(s, "The Duration Confound");
  dividerLine(s, 1.38);

  s.addText("Is GPT-4o reading language — or transcript length?", {
    x: 0.35, y: 1.5, w: 9.3, h: 0.5,
    fontSize: 18, bold: true, color: NAVY, fontFace: "Calibri", margin: 0
  });

  // Two columns
  // Left: the concern
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.35, y: 2.1, w: 4.45, h: 2.8,
    fill: { color: WHITE }, line: { color: LT_GRAY }, shadow: mkShadow()
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.35, y: 2.1, w: 4.45, h: 0.38,
    fill: { color: AMBER }, line: { color: AMBER }
  });
  s.addText("The concern", {
    x: 0.45, y: 2.1, w: 4.25, h: 0.38,
    fontSize: 12, bold: true, color: WHITE, fontFace: "Calibri", valign: "middle", margin: 0
  });
  const concern = [
    "A call still running at 260s already signals 'probably a sale' independent of content",
    "No-sale median is 163s — most no-sales end before any window",
    "GPT-4o sees longer transcripts for sales → may learn length, not language",
    "AUC 0.900 at 400s may partly reflect survivor selection, not semantic prediction",
  ];
  concern.forEach((c, i) => {
    s.addText([{ text: c, options: { bullet: true } }], {
      x: 0.45, y: 2.55 + i * 0.52, w: 4.2, h: 0.48,
      fontSize: 11, color: NAVY, fontFace: "Calibri", margin: 0
    });
  });

  // Right: the test
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: 2.1, w: 4.45, h: 2.8,
    fill: { color: WHITE }, line: { color: TEAL }, shadow: mkShadow()
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: 2.1, w: 4.45, h: 0.38,
    fill: { color: TEAL }, line: { color: TEAL }
  });
  s.addText("The next experiment", {
    x: 5.3, y: 2.1, w: 4.25, h: 0.38,
    fontSize: 12, bold: true, color: WHITE, fontFace: "Calibri", valign: "middle", margin: 0
  });
  const test = [
    "Build a duration-ratio baseline: predict sale if duration_so_far / avg_duration > threshold",
    "Compare its conditional AUC against GPT-4o at each window",
    "If AUCs are similar → model is proxying duration, not language",
    "Fix: truncate prompts to fixed token length to eliminate length leakage",
  ];
  test.forEach((t, i) => {
    s.addText([{ text: t, options: { bullet: true } }], {
      x: 5.3, y: 2.55 + i * 0.52, w: 4.2, h: 0.48,
      fontSize: 11, color: NAVY, fontFace: "Calibri", margin: 0
    });
  });
}

// ──────────────────────────────────────────────────────────────────────────────
// SLIDE 15 — OPEN QUESTIONS
// ──────────────────────────────────────────────────────────────────────────────
{
  const s = lightSlide(pres);
  addLeftAccent(s);
  slideLabel(s, "Act 4 · Open Questions");
  slideTitle(s, "Four Questions for Future Work");
  dividerLine(s, 1.38);

  const qs = [
    {
      n: "Q1", color: TEAL,
      q: "How much of the AUC is duration vs. language?",
      a: "Duration-ratio baseline needed. Fixed-token prompt ablation would isolate the linguistic contribution.",
    },
    {
      n: "Q2", color: NAVY,
      q: "Does fine-tuning change the threshold where signal emerges?",
      a: "Zero-shot GPT-4o is a hard test. Fine-tuning on domain-specific data may push the AUC threshold window earlier.",
    },
    {
      n: "Q3", color: AMBER,
      q: "Is backward induction the right framework for stochastic real-world deployment?",
      a: "Exact here (known reward, finite horizon). RL or bandit approaches warranted when reward structure is itself uncertain.",
    },
    {
      n: "Q4", color: MID_GRAY,
      q: "What dataset size is required for reliable policy evaluation?",
      a: "12-call test set is directional at best. Power analysis suggests 80+ test conversations per domain for ±10% sales estimates.",
    },
  ];

  qs.forEach((q, i) => {
    const row = Math.floor(i / 2);
    const col = i % 2;
    const bx = 0.35 + col * 4.75;
    const by = 1.5 + row * 2.0;
    s.addShape(pres.shapes.RECTANGLE, {
      x: bx, y: by, w: 4.5, h: 1.75,
      fill: { color: WHITE }, line: { color: LT_GRAY }, shadow: mkShadow()
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: bx, y: by, w: 4.5, h: 0.10,
      fill: { color: q.color }, line: { color: q.color }
    });
    s.addText(q.n + "  " + q.q, {
      x: bx + 0.15, y: by + 0.18, w: 4.2, h: 0.58,
      fontSize: 11, bold: true, color: NAVY, fontFace: "Calibri", margin: 0
    });
    s.addText(q.a, {
      x: bx + 0.15, y: by + 0.82, w: 4.2, h: 0.82,
      fontSize: 10.5, color: MID_GRAY, fontFace: "Calibri", margin: 0
    });
  });
}

// ──────────────────────────────────────────────────────────────────────────────
// SLIDE 16 — CONCLUSION (dark)
// ──────────────────────────────────────────────────────────────────────────────
{
  const s = darkSlide(pres);
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.10, h: 5.625,
    fill: { color: TEAL }, line: { color: TEAL }
  });

  slideLabel(s, "Conclusion", true);
  s.addText("What We Know", {
    x: 0.35, y: 0.55, w: 9.3, h: 0.65,
    fontSize: 28, bold: true, color: WHITE, fontFace: "Calibri", margin: 0
  });
  dividerLine(s, 1.28, true);

  const takeaways = [
    { icon: "01", text: "The backward induction + LLM logprobs framework generalizes in structure, not in parameters. Windows and cost/benefit must be re-calibrated per domain." },
    { icon: "02", text: "Conditional AUC is the correct discriminability metric for survival-indexed classifiers. Unconditional AUC contaminates the estimate with calls the agent never acts on." },
    { icon: "03", text: "AUC > ~0.65 at the first window is the empirical threshold for useful stopping policy. Below this, tuning cannot rescue the agent." },
    { icon: "04", text: "The 350s mark in dealership calls is a linguistic dead zone. Mid-negotiation language is ambiguous — the agent's rational response is to do nothing, which it correctly does." },
    { icon: "05", text: "The duration confound is unresolved. High AUC at late windows may reflect call survival rather than semantic understanding. This is the critical open question." },
  ];

  takeaways.forEach((t, i) => {
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.35, y: 1.38 + i * 0.79, w: 0.45, h: 0.58,
      fill: { color: TEAL, transparency: 30 }, line: { color: TEAL, transparency: 30 }
    });
    s.addText(t.icon, {
      x: 0.35, y: 1.38 + i * 0.79, w: 0.45, h: 0.58,
      fontSize: 11, bold: true, color: TEAL_LT, fontFace: "Calibri", align: "center", valign: "middle", margin: 0
    });
    s.addText(t.text, {
      x: 0.9, y: 1.42 + i * 0.79, w: 8.75, h: 0.55,
      fontSize: 11, color: "CBD8E8", fontFace: "Calibri", valign: "middle", margin: 0
    });
  });
}

// ──────────────────────────────────────────────────────────────────────────────
// WRITE
// ──────────────────────────────────────────────────────────────────────────────
pres.writeFile({ fileName: "optimal_stopping_seminar.pptx" })
  .then(() => console.log("✓ optimal_stopping_seminar.pptx written"))
  .catch(err => { console.error(err); process.exit(1); });
