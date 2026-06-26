'use strict';

// ─── Constants ───────────────────────────────────────────────────────────────
const DEFAULT_HP             = 8;
const VCV_VERSION            = '2.5.2';
const CONNECTION_BREAK_PENALTY = 1_000_000;
const TARGET_ROW_HP            = 100;

// ─── Module width lookup (populated at load from module_widths.json) ─────────
let moduleWidths = {};

function moduleHp(slug) {
  return moduleWidths[slug] ?? DEFAULT_HP;
}

function autoRows(slots) {
  const totalHp = slots.reduce((sum, [, slug]) => sum + moduleHp(normalizeSlug(slug)), 0);
  return Math.max(1, Math.ceil(totalHp / TARGET_ROW_HP));
}

// ─── Param range table (populated at load from param_ranges.json) ────────────
let paramRanges = {};

// ─── Brand aliases (populated at load from brand_aliases.json) ───────────────
// Maps alternate brand slugs to canonical VCV slugs.
// e.g., "JWModules" → "JW-Modules"
let brandAliases = {};

// ─── Conversion helpers ───────────────────────────────────────────────────────

function rgb565ToHex(c) {
  const r = Math.floor(((c >> 11) & 0x1F) * 255 / 31);
  const g = Math.floor(((c >>  5) & 0x3F) * 255 / 63);
  const b = Math.floor(( c        & 0x1F) * 255 / 31);
  return '#' + [r, g, b].map(v => v.toString(16).padStart(2, '0')).join('');
}

function parseSlug(slug) {
  const i = slug.indexOf(':');
  if (i !== -1) {
    let plugin = slug.slice(0, i).trim();
    const model = slug.slice(i + 1).trim();
    if (plugin === 'RackCore') plugin = 'Core';
    // Apply brand alias mapping (e.g., "JWModules" → "JW-Modules")
    if (brandAliases[plugin]) plugin = brandAliases[plugin];
    return [plugin, model];
  }
  // Bare slug (no plugin prefix) — assume 4msCompany built-in
  return ['4msCompany', slug];
}

function normalizeSlug(slug) {
  // Airwindows modules in MetaModule are separate (e.g. Airwindows:Galactic), but
  // in VCV they're all one module (Airwin2Rack:Airwin2Rack) with the effect selected via data.
  if (slug.startsWith('Airwindows:')) return 'Airwin2Rack:Airwin2Rack';

  // Apply brand alias to the plugin prefix (e.g., "JWModules:GridSeq" → "JW-Modules:GridSeq")
  const i = slug.indexOf(':');
  if (i !== -1) {
    const plugin = slug.slice(0, i);
    const model = slug.slice(i + 1);
    const canonical = brandAliases[plugin] || plugin;
    return canonical + ':' + model;
  }
  return '4msCompany:' + slug;
}

// Param value scaling: MetaModule saves all params as normalized 0-1 values
// (via ParamQuantity::getScaledValue()). VCV Rack loads params as native values.
// For 4ms built-in modules, param_ranges.json provides the [min, max] ranges
// so we can scale: native = normalized × (max − min) + min.
// For other modules (no range table), values with [0,1] native range are
// correct as-is; other ranges remain unscaled (unavoidable without VCV runtime).

function buildParams(slotIdx, pByM, slug) {
  const pm = pByM[slotIdx] || {};
  const ranges = paramRanges[slug] || {};
  return Object.keys(pm)
    .map(Number)
    .sort((a, b) => a - b)
    .map(pid => {
      let value = pm[pid];
      const r = ranges[pid];
      if (r) value = value * (r[1] - r[0]) + r[0];
      return { id: pid, value };
    });
}

function tryParseJson(text) {
  try { return JSON.parse(text); } catch { return null; }
}

// Derive Hub knob positions from mapped param values.
// MetaModule doesn't save hub knob values to static_knobs, so we back-calculate
// from each mapped param's normalized value and the mapping's RangeMin/RangeMax.
// hub_knob = clamp((param_value - RangeMin) / (RangeMax - RangeMin), 0, 1)
function buildHubParams(pd, pByM) {
  const hubParams = {};
  for (const ks of (pd.mapped_knobs || [])) {
    for (const mk of (ks.set || [])) {
      if (hubParams[mk.panel_knob_id] !== undefined) continue; // first mapping wins
      const paramVal = (pByM[mk.module_id] || {})[mk.param_id];
      if (paramVal === undefined) continue;
      const rMin = mk.min ?? 0;
      const rMax = mk.max ?? 1;
      const hubVal = rMax !== rMin
        ? Math.max(0, Math.min(1, (paramVal - rMin) / (rMax - rMin)))
        : 0;
      hubParams[mk.panel_knob_id] = hubVal;
    }
  }
  return Object.keys(hubParams).map(Number).sort((a, b) => a - b)
    .map(id => ({ id, value: hubParams[id] }));
}

function buildHubMappings(pd, hubSlot) {
  const NUM_SETS = 8;
  const knobSets = (pd.mapped_knobs || []).slice(0, NUM_SETS);
  const mappings = knobSets.map(ks =>
    (ks.set || []).map(mk => ({
      DstModID: mk.module_id,
      DstObjID: mk.param_id,
      SrcModID: hubSlot,
      SrcObjID: mk.panel_knob_id,
      RangeMin: mk.min ?? 0,
      RangeMax: mk.max ?? 1,
      CurveType: mk.curve_type ?? 0,
      AliasName: '',
    }))
  );
  const knobSetNames = knobSets.map(ks => ks.name || '');
  while (mappings.length < NUM_SETS) { mappings.push([]); knobSetNames.push(''); }
  return { Mappings: mappings, KnobSetNames: knobSetNames };
}

function computeRowAssignments(slots, intCables, numRows) {
  const n = slots.length;
  if (n === 0) return [];
  numRows = Math.min(numRows, n);

  const hps    = slots.map(([, slug]) => moduleHp(normalizeSlug(slug)));
  const prefix = new Array(n + 1).fill(0);
  for (let i = 0; i < n; i++) prefix[i + 1] = prefix[i] + hps[i];
  const target = prefix[n] / numRows;

  // Build set of adjacent order-index pairs sharing a cable.
  const slotToOrder = {};
  slots.forEach(([idx], i) => { slotToOrder[parseInt(idx)] = i; });
  const connAdj = new Set();
  for (const cable of (intCables || [])) {
    const a = slotToOrder[cable.out.module_id];
    for (const inJack of (cable.ins || [])) {
      const b = slotToOrder[inJack.module_id];
      if (a !== undefined && b !== undefined && Math.abs(a - b) === 1) {
        connAdj.add(`${Math.min(a, b)},${Math.max(a, b)}`);
      }
    }
  }

  const breakPenalty = at => connAdj.has(`${at - 1},${at}`) ? CONNECTION_BREAK_PENALTY : 0;
  const widthCost = (s, e) => { const w = prefix[e] - prefix[s]; return (w - target) ** 2; };

  // dp[j][i] = min cost to place first i modules in j rows
  const dp  = Array.from({ length: numRows + 1 }, () => new Array(n + 1).fill(Infinity));
  const par = Array.from({ length: numRows + 1 }, () => new Array(n + 1).fill(-1));
  dp[0][0] = 0;

  for (let j = 1; j <= numRows; j++) {
    for (let i = j; i <= n; i++) {
      for (let k = j - 1; k < i; k++) {
        const pen  = k > 0 ? breakPenalty(k) : 0;
        const cost = dp[j - 1][k] + widthCost(k, i) + pen;
        if (cost < dp[j][i]) { dp[j][i] = cost; par[j][i] = k; }
      }
    }
  }

  // Backtrack to recover row start indices.
  const rowStarts = [];
  let i = n;
  for (let j = numRows; j > 0; j--) {
    const k = par[j][i];
    if (k > 0) rowStarts.push(k);
    i = k;
  }
  rowStarts.reverse();
  rowStarts.unshift(0);

  const assignments = new Array(n).fill(0);
  for (let r = 0; r < rowStarts.length; r++) {
    const start = rowStarts[r];
    const end   = r + 1 < rowStarts.length ? rowStarts[r + 1] : n;
    for (let ii = start; ii < end; ii++) assignments[ii] = r;
  }
  return assignments;
}

function convert(yamlText, numRows) {
  // Users sometimes prefix patch names with '!' (e.g. "!_airwintests").
  // YAML treats bare '!' as a tag indicator, so pre-quote any unquoted scalar
  // values that start with '!' (but not '!!' which are valid YAML type tags).
  yamlText = yamlText.replace(/(:\s*)(![^!\n][^\n]*)/g, (_, prefix, val) =>
    `${prefix}"${val.trimEnd().replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`
  );
  const data = jsyaml.load(yamlText);
  if (!data || !data.PatchData) throw new Error('Not a valid MetaModule .yml patch file.');

  const pd        = data.PatchData;
  const patchName = pd.patch_name || 'Untitled';
  const patchDesc = pd.description || '';

  const rawSlugs = pd.module_slugs || {};
  const slots    = Object.entries(rawSlugs).sort((a, b) => parseInt(a[0]) - parseInt(b[0]));

  // params: {module_id: {param_id: value}}
  const pByM = {};
  for (const sk of (pd.static_knobs || [])) {
    if (!pByM[sk.module_id]) pByM[sk.module_id] = {};
    pByM[sk.module_id][sk.param_id] = sk.value;
  }

  // module states: {module_id: parsed_json_or_string}
  // rawsByM stores the original JSON string for valid-JSON states so we can
  // inject it verbatim after JSON.stringify, preserving "2.0" vs "2" precision
  // (JSON.parse collapses 2.0 → JS integer 2, losing the decimal VCV needs).
  const sByM = {};
  const rawsByM = {};
  for (const ms of (pd.vcvModuleStates || [])) {
    const raw = ms.data || '';
    if (!raw) continue;
    const parsed = tryParseJson(raw);
    if (parsed !== null) {
      rawsByM[ms.module_id] = raw;  // keep original for verbatim injection
      sByM[ms.module_id] = '__RAW_STATE__';  // placeholder
    } else {
      // Plain string (e.g. a file path for BWAVP) — serialize normally.
      sByM[ms.module_id] = raw;
    }
  }

  const bypassed  = new Set(pd.bypassed_modules || []);
  const intCables = pd.int_cables || [];

  const rowAssign = computeRowAssignments(slots, intCables, numRows ?? autoRows(slots));
  const xByRow    = {};
  const vcvModules = [];

  slots.forEach(([rawKey, rawSlug], oi) => {
    const sid          = parseInt(rawKey);
    const slug         = normalizeSlug(rawSlug);
    const [plugin, model] = parseSlug(slug);
    const row          = rowAssign[oi];

    if (xByRow[row] === undefined) xByRow[row] = 0;
    const xPos  = xByRow[row];
    xByRow[row] = xPos + moduleHp(slug);

    const mod = {
      id: sid, plugin, version: '2.0.0', model,
      params: sid === 0 ? buildHubParams(pd, pByM) : buildParams(sid, pByM, slug),
      pos: [xPos, row],
    };

    // Adjacency links — only within the same row.
    if (oi > 0 && rowAssign[oi - 1] === row)
      mod.leftModuleId  = parseInt(slots[oi - 1][0]);
    if (oi < slots.length - 1 && rowAssign[oi + 1] === row)
      mod.rightModuleId = parseInt(slots[oi + 1][0]);

    if (sid === 0) {
      const hubData = { PatchName: patchName, PatchDesc: patchDesc, ...buildHubMappings(pd, 0) };
      mod.data = hubData;
    } else if (rawsByM[sid] !== undefined) {
      mod.data = `__RAW_STATE_${sid}__`;  // replaced verbatim after stringify
    } else if (sByM[sid] !== undefined) {
      mod.data = sByM[sid];
    } else if (rawSlug.startsWith('Airwindows:')) {
      // No saved state blob, but the slug names the effect (e.g.
      // Airwindows:BitShiftGain). All Airwindows map to one VCV module whose FX
      // is chosen via data; without this, Airwin2Rack defaults to Galactic.
      mod.data = {
        airwindowSelectedFX: rawSlug.slice(rawSlug.indexOf(':') + 1),
        polyphonyMode: 0, lockedType: false, randomizeFX: false, blockSize: 4,
      };
    }

    if (bypassed.has(sid)) mod.bypass = true;
    vcvModules.push(mod);
  });

  // Build cables — fan-out: one YML cable → one VCV entry per input.
  const vcvCables = [];
  let cableId = 1000;
  for (const cable of intCables) {
    const rawColor = cable.color;
    const hexColor = typeof rawColor === 'number'
      ? rgb565ToHex(rawColor)
      : (rawColor || rgb565ToHex(61865));
    for (const inJack of (cable.ins || [])) {
      vcvCables.push({
        id: cableId++,
        outputModuleId: cable.out.module_id,
        outputId:       cable.out.jack_id,
        inputModuleId:  inJack.module_id,
        inputId:        inJack.jack_id,
        color:          hexColor,
      });
    }
  }

  // mapped_outs: module output → Hub input (panel output jack)
  for (const mo of (pd.mapped_outs || [])) {
    if (!mo.out) continue;
    vcvCables.push({
      id: cableId++,
      outputModuleId: mo.out.module_id,
      outputId:       mo.out.jack_id,
      inputModuleId:  0,
      inputId:        mo.panel_jack_id,
      color:          '#f6a623',
    });
  }

  // mapped_ins: Hub output (panel input jack) → module inputs (fan-out)
  for (const mi of (pd.mapped_ins || [])) {
    for (const inJack of (mi.ins || [])) {
      vcvCables.push({
        id: cableId++,
        outputModuleId: 0,
        outputId:       mi.panel_jack_id,
        inputModuleId:  inJack.module_id,
        inputId:        inJack.jack_id,
        color:          '#f6a623',
      });
    }
  }

  // Append AudioInterface2 if the patch suggests a sample rate or block size.
  const suggestedSr = pd.suggested_samplerate || 0;
  const suggestedBs = pd.suggested_blocksize  || 0;
  if (suggestedSr || suggestedBs) {
    const audioXPos = xByRow[0] || 0;
    vcvModules.push({
      id: 9999,
      plugin: 'Core',
      version: '2.6.6',
      model: 'AudioInterface2',
      params: [{ id: 0, value: 1.0 }],
      data: {
        audio: {
          driver: 5,
          sampleRate: suggestedSr || 0.0,
          blockSize:  suggestedBs || 0,
          inputOffset: 0,
          outputOffset: 0,
        },
        dcFilter: true,
      },
      pos: [audioXPos, 0],
    });
  }

  return { patch: { version: VCV_VERSION, modules: vcvModules, cables: vcvCables }, rawsByM };
}

// ─── Load lookup tables ───────────────────────────────────────────────────────
fetch('param_ranges.json')
  .then(r => r.ok ? r.json() : Promise.reject())
  .then(data => { paramRanges = data; })
  .catch(() => {});  // silently ignore (e.g. file:// context)

fetch('module_widths.json')
  .then(r => r.ok ? r.json() : Promise.reject())
  .then(data => { moduleWidths = data; })
  .catch(() => {});  // silently ignore (e.g. file:// context)

fetch('brand_aliases.json')
  .then(r => r.ok ? r.json() : Promise.reject())
  .then(data => { brandAliases = data; })
  .catch(() => {});  // silently ignore (e.g. file:// context)

// ─── UI ───────────────────────────────────────────────────────────────────────

const dropZone    = document.getElementById('dropZone');
const fileInput   = document.getElementById('fileInput');
const fileBadge   = document.getElementById('fileBadge');
const rowCount    = document.getElementById('rowCount');
const downloadArea = document.getElementById('downloadArea');
const downloadName = document.getElementById('downloadName');
const downloadLink = document.getElementById('downloadLink');
const errorArea   = document.getElementById('errorArea');

let currentFile = null;
let currentText = null;
let blobUrl     = null;

function showError(msg) {
  errorArea.textContent = msg;
  errorArea.classList.add('visible');
  downloadArea.classList.remove('visible');
}

function hideMessages() {
  errorArea.classList.remove('visible');
  downloadArea.classList.remove('visible');
}

function setFile(file) {
  if (!/\.ya?ml$/i.test(file.name)) {
    showError('Not a valid .yml file.');
    return;
  }
  currentFile = file;
  fileBadge.textContent = file.name;
  fileBadge.classList.add('visible');
  hideMessages();

  const reader = new FileReader();
  reader.onload  = e => { currentText = e.target.result; runConvert(); };
  reader.onerror = () => showError('Could not read the file.');
  reader.readAsText(file);
}

function runConvert() {
  if (!currentText) return;
  hideMessages();

  try {
    const numRows = rowCount.value === 'auto' ? null : parseInt(rowCount.value);
    const { patch, rawsByM } = convert(currentText, numRows);
    // VCV Rack and the Hub use json_real_value() (jansson) to read numeric
    // fields, which returns 0.0 for JSON integers (no decimal point).
    // JS JSON.stringify serialises whole-number floats as integers (2.0 → 2),
    // so we must append ".0" to any integer-valued numeric field that VCV
    // expects to be a real: param values, RangeMin, RangeMax.
    let json = JSON.stringify(patch, null, 2)
      .replace(/"value": (-?\d+)(?![\.\d])/g,   '"value": $1.0')
      .replace(/"RangeMin": (-?\d+)(?![\.\d])/g, '"RangeMin": $1.0')
      .replace(/"RangeMax": (-?\d+)(?![\.\d])/g, '"RangeMax": $1.0');
    // Inject raw module state strings verbatim to preserve float precision
    // (e.g. "2.0" stays "2.0" instead of being collapsed to integer "2").
    for (const [sid, raw] of Object.entries(rawsByM)) {
      json = json.replace(`"data": "__RAW_STATE_${sid}__"`, `"data": ${raw}`);
    }

    // Revoke previous blob to avoid memory leak.
    if (blobUrl) URL.revokeObjectURL(blobUrl);
    blobUrl = URL.createObjectURL(new Blob([json], { type: 'application/json' }));

    const outName = currentFile.name.replace(/\.ya?ml$/i, '') + '.vcv';
    downloadName.textContent = outName;
    downloadLink.href        = blobUrl;
    downloadLink.download    = outName;
    downloadArea.classList.remove('visible');
    void downloadArea.offsetWidth; // force reflow to replay animation
    downloadArea.classList.add('visible');
  } catch (err) {
    showError(err.message);
  }
}

fileInput.addEventListener('change', e => {
  if (e.target.files[0]) setFile(e.target.files[0]);
});

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});

rowCount.addEventListener('change', () => { if (currentText) runConvert(); });
