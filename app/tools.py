from dataclasses import dataclass, field


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict
    required: list[str] = field(default_factory=list)
    execution_domain: str = "frontend"  # "frontend" | "server"


class ToolRegistry:
    """Central registry for all tools. Provides schema generation in OpenAI and Claude formats."""

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def register_many(self, tools: list[ToolDefinition]) -> None:
        for t in tools:
            self.register(t)

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def all_schemas_openai(self) -> list[dict]:
        result = []
        for t in self._tools.values():
            schema = {**t.parameters}
            if t.required:
                schema["required"] = t.required
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": schema,
                    },
                }
            )
        return result

    def all_schemas_claude(self) -> list[dict]:
        result = []
        for t in self._tools.values():
            schema = {**t.parameters}
            if t.required:
                schema["required"] = t.required
            result.append(
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": schema,
                }
            )
        return result

    @property
    def server_tools(self) -> list[str]:
        return [
            name for name, t in self._tools.items() if t.execution_domain == "server"
        ]


# ─── L1: Query tools (read-only) ─────────────────────────────────────────────

QUERY_TOOLS = [
    ToolDefinition(
        name="get_molecule_info",
        description="Get molecule overview: total atoms, element counts, bond count, selection. Use as first step to understand the loaded structure.",
        parameters={"type": "object", "properties": {}},
    ),
    ToolDefinition(
        name="get_atom_info",
        description="Get position (x,y,z) and element for specific atoms. Returns coordinates in Angstroms.",
        parameters={
            "type": "object",
            "properties": {
                "indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Atom indices (0-based)",
                }
            },
        },
        required=["indices"],
    ),
    ToolDefinition(
        name="get_bonded_atoms",
        description="Get bond connectivity for atoms. Returns bonded atom indices and elements. Essential for understanding topology before scans or edits.",
        parameters={
            "type": "object",
            "properties": {
                "indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Atom indices to query. If omitted, uses current selection.",
                }
            },
        },
    ),
    ToolDefinition(
        name="measure_distance",
        description="Measure distance between 2 atoms in Angstroms. Returns distance_angstrom. Can specify indices directly (preferred) or use selection.",
        parameters={
            "type": "object",
            "properties": {
                "atom1": {
                    "type": "integer",
                    "description": "First atom index (0-based)",
                },
                "atom2": {
                    "type": "integer",
                    "description": "Second atom index (0-based)",
                },
            },
        },
    ),
    ToolDefinition(
        name="measure_angle",
        description="Measure angle formed by 3 atoms in degrees (atom2 is vertex). Returns angle_degrees. Can specify indices directly (preferred) or use selection.",
        parameters={
            "type": "object",
            "properties": {
                "atom1": {"type": "integer", "description": "First atom (0-based)"},
                "atom2": {"type": "integer", "description": "Vertex atom (0-based)"},
                "atom3": {"type": "integer", "description": "Third atom (0-based)"},
            },
        },
    ),
    ToolDefinition(
        name="measure_dihedral",
        description="Measure dihedral/torsion angle between 4 atoms in degrees. Returns dihedral_degrees. Can specify indices directly (preferred) or use selection.",
        parameters={
            "type": "object",
            "properties": {
                "atom1": {"type": "integer", "description": "First atom (0-based)"},
                "atom2": {"type": "integer", "description": "Second atom (0-based)"},
                "atom3": {"type": "integer", "description": "Third atom (0-based)"},
                "atom4": {"type": "integer", "description": "Fourth atom (0-based)"},
            },
        },
    ),
    ToolDefinition(
        name="get_cached_energies",
        description="Retrieve cached energy results from last calculate_all_energies/optimize_geometry/run_md. Avoids recalculation. Follow with create_chart.",
        parameters={"type": "object", "properties": {}},
    ),
    ToolDefinition(
        name="read_file",
        description="Read text content of a file in the open folder.",
        parameters={
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Filename to read"}
            },
        },
        required=["filename"],
    ),
    ToolDefinition(
        name="list_folder_files",
        description="List all files in the currently open folder.",
        parameters={"type": "object", "properties": {}},
    ),
    ToolDefinition(
        name="web_search",
        description="Search the web for chemistry info, properties, SMILES, safety data, reactions. Returns answer and source snippets. Use for anything you don't know.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "search_depth": {
                    "type": "string",
                    "enum": ["basic", "advanced"],
                    "description": "basic (fast) or advanced (thorough). Default: basic",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Results count (1-10, default: 5)",
                },
                "topic": {
                    "type": "string",
                    "enum": ["general", "news"],
                    "description": "Default: general",
                },
            },
        },
        required=["query"],
        execution_domain="server",
    ),
]

# ─── L2: Selection tools ─────────────────────────────────────────────────────

SELECTION_TOOLS = [
    ToolDefinition(
        name="select_atoms",
        description="Select atoms by 0-based indices. Sets context for edit tools (remove_atoms, change_atom_element, set_bond_distance, set_angle, set_dihedral_angle). Use add:true to extend.",
        parameters={
            "type": "object",
            "properties": {
                "indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Atom indices to select",
                },
                "add": {
                    "type": "boolean",
                    "description": "If true, add to current selection",
                },
            },
        },
        required=["indices"],
    ),
    ToolDefinition(
        name="select_atoms_by_element",
        description="Select all atoms of an element type (e.g. C, O, N). Faster than listing indices. Use add:true to combine with existing selection.",
        parameters={
            "type": "object",
            "properties": {
                "element": {"type": "string", "description": "Element symbol"},
                "add": {
                    "type": "boolean",
                    "description": "If true, add to current selection",
                },
            },
        },
        required=["element"],
    ),
    ToolDefinition(
        name="select_all_atoms",
        description="Select every atom in the molecule.",
        parameters={"type": "object", "properties": {}},
    ),
    ToolDefinition(
        name="select_connected",
        description="Expand selection to include atoms directly bonded to currently selected atoms.",
        parameters={"type": "object", "properties": {}},
    ),
    ToolDefinition(
        name="clear_selection",
        description="Clear all atom selections.",
        parameters={"type": "object", "properties": {}},
    ),
]

# ─── L3: Edit tools ──────────────────────────────────────────────────────────

EDIT_TOOLS = [
    ToolDefinition(
        name="add_atom",
        description="Add atom at coordinates or bonded to selected atom. Use bondToSelected:true with 1 atom selected for automatic positioning.",
        parameters={
            "type": "object",
            "properties": {
                "element": {
                    "type": "string",
                    "description": "Element symbol (e.g. C, H, O)",
                },
                "x": {
                    "type": "number",
                    "description": "X coordinate (optional if bondToSelected)",
                },
                "y": {"type": "number", "description": "Y coordinate"},
                "z": {"type": "number", "description": "Z coordinate"},
                "bondToSelected": {
                    "type": "boolean",
                    "description": "Bond to selected atom at typical bond length",
                },
            },
        },
        required=["element"],
    ),
    ToolDefinition(
        name="remove_atoms",
        description="Delete all currently selected atoms. Requires selection first.",
        parameters={"type": "object", "properties": {}},
    ),
    ToolDefinition(
        name="change_atom_element",
        description="Change element of selected atoms. Requires selection. Use select_atoms first.",
        parameters={
            "type": "object",
            "properties": {
                "element": {"type": "string", "description": "New element symbol"}
            },
        },
        required=["element"],
    ),
    ToolDefinition(
        name="set_bond_distance",
        description="Set exact distance between 2 selected atoms in Angstroms. Moves the smaller fragment. Requires exactly 2 atoms selected.",
        parameters={
            "type": "object",
            "properties": {
                "distance": {
                    "type": "number",
                    "description": "Target distance in Angstroms",
                }
            },
        },
        required=["distance"],
    ),
    ToolDefinition(
        name="set_angle",
        description="Set bond angle for 3 selected atoms to exact value. B is vertex (A-B-C). Rotates fragment on A side. Requires exactly 3 atoms selected.",
        parameters={
            "type": "object",
            "properties": {
                "angle": {
                    "type": "number",
                    "description": "Target angle in degrees (0-180)",
                }
            },
        },
        required=["angle"],
    ),
    ToolDefinition(
        name="set_dihedral_angle",
        description="Set dihedral for 4 selected atoms. Rotates fragment on D side around B-C axis (A-B-C-D). Requires exactly 4 atoms selected.",
        parameters={
            "type": "object",
            "properties": {
                "angle": {
                    "type": "number",
                    "description": "Target dihedral in degrees (0-360)",
                }
            },
        },
        required=["angle"],
    ),
    ToolDefinition(
        name="transform_atoms",
        description="Rotate or translate atoms around an axis. Specify axisAtom1, axisAtom2, atomsToMove, and either angle (degrees) or distance (Angstroms). Use get_bonded_atoms or split_molecule to identify fragment indices.",
        parameters={
            "type": "object",
            "properties": {
                "axisAtom1": {
                    "type": "integer",
                    "description": "First axis atom (0-based)",
                },
                "axisAtom2": {
                    "type": "integer",
                    "description": "Second axis atom (0-based)",
                },
                "atomsToMove": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Atom indices to move (0-based)",
                },
                "angle": {
                    "type": "number",
                    "description": "Rotation degrees (use this OR distance)",
                },
                "distance": {
                    "type": "number",
                    "description": "Translation Angstroms (use this OR angle)",
                },
            },
        },
        required=["axisAtom1", "axisAtom2", "atomsToMove"],
    ),
    ToolDefinition(
        name="split_molecule",
        description="Split molecule by breaking bond. Returns fragment1[] and fragment2[] with atom indices. Use smaller fragment as atomsToMove for rotational_scan or translation_scan.",
        parameters={
            "type": "object",
            "properties": {
                "atom1": {"type": "integer", "description": "First atom (0-based)"},
                "atom2": {
                    "type": "integer",
                    "description": "Second atom (0-based, bonded to atom1)",
                },
            },
        },
        required=["atom1", "atom2"],
    ),
    ToolDefinition(
        name="add_hydrogens",
        description="Add missing hydrogen atoms to entire molecule using standard valence rules.",
        parameters={
            "type": "object",
            "properties": {
                "pH": {
                    "type": "number",
                    "description": "pH for protonation state (default 7.4)",
                }
            },
        },
    ),
]

# ─── L4: Generation tools ────────────────────────────────────────────────────

GENERATION_TOOLS = [
    ToolDefinition(
        name="rotational_scan",
        description="Torsion scan: generate frames by rotating fragment around axis. Returns frameCount. Auto-picks smaller fragment. Not for rings. Follow with calculate_all_energies then create_chart.",
        parameters={
            "type": "object",
            "properties": {
                "axisAtom1": {
                    "type": "integer",
                    "description": "First axis atom (0-based)",
                },
                "axisAtom2": {
                    "type": "integer",
                    "description": "Second axis atom (0-based)",
                },
                "atomsToMove": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Override: atoms to rotate. If omitted, auto-picks smaller fragment.",
                },
                "increment": {
                    "type": "number",
                    "description": "Step size in degrees (default: 10)",
                },
                "startAngle": {
                    "type": "number",
                    "description": "Start angle (default: 0)",
                },
                "endAngle": {
                    "type": "number",
                    "description": "End angle (default: 360)",
                },
            },
        },
        required=["axisAtom1", "axisAtom2"],
    ),
    ToolDefinition(
        name="translation_scan",
        description="Dissociation scan: translate fragment along axis in distance increments. Returns frameCount. Follow with calculate_all_energies then create_chart.",
        parameters={
            "type": "object",
            "properties": {
                "axisAtom1": {
                    "type": "integer",
                    "description": "First axis atom (0-based)",
                },
                "axisAtom2": {
                    "type": "integer",
                    "description": "Second axis atom (0-based)",
                },
                "atomsToMove": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Atoms to translate (0-based)",
                },
                "startDistance": {
                    "type": "number",
                    "description": "Start distance in Angstroms (default: 0)",
                },
                "endDistance": {
                    "type": "number",
                    "description": "End distance (default: 3)",
                },
                "increment": {
                    "type": "number",
                    "description": "Step size in Angstroms (default: 0.2)",
                },
            },
        },
        required=["axisAtom1", "axisAtom2", "atomsToMove"],
    ),
    ToolDefinition(
        name="angle_scan",
        description="Angle scan: rotate fragment through angle range around pivot atom2 (A-B-C). Returns frameCount. Follow with calculate_all_energies then create_chart.",
        parameters={
            "type": "object",
            "properties": {
                "atom1": {"type": "integer", "description": "First atom (0-based)"},
                "atom2": {"type": "integer", "description": "Pivot atom (0-based)"},
                "atom3": {"type": "integer", "description": "Third atom (0-based)"},
                "atomsToMove": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Atoms to rotate (0-based)",
                },
                "increment": {
                    "type": "number",
                    "description": "Step degrees (default: 10)",
                },
                "startAngle": {"type": "number", "description": "Start (default: 0)"},
                "endAngle": {"type": "number", "description": "End (default: 360)"},
            },
        },
        required=["atom1", "atom2", "atom3"],
    ),
    ToolDefinition(
        name="calculate_energy",
        description="Single-point MACE energy for current geometry. Returns energy_eV and per-atom forces by default (needed for toggle_force_arrows). Always specify model.",
        parameters={
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "enum": ["mace-mp-0a", "mace-mp-0b3", "mace-mpa-0"],
                    "description": "MACE model",
                },
                "includeForces": {
                    "type": "boolean",
                    "description": "Include per-atom forces (default: true)",
                },
            },
        },
        required=["model"],
        execution_domain="server",
    ),
    ToolDefinition(
        name="calculate_all_energies",
        description="Batch MACE energy for all frames. Returns energies array with scanXValues for charting. Required after any scan. Follow with create_chart. Always specify model.",
        parameters={
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "enum": ["mace-mp-0a", "mace-mp-0b3", "mace-mpa-0"],
                    "description": "mace-mp-0a (fast), mace-mp-0b3 (high-P), mace-mpa-0 (accurate)",
                },
                "includeForces": {
                    "type": "boolean",
                    "description": "Include per-atom forces (default: true)",
                },
            },
        },
        required=["model"],
        execution_domain="server",
    ),
    ToolDefinition(
        name="optimize_geometry",
        description="MACE geometry optimization. Returns converged, steps, energy_eV, trajectory, and per-atom forces by default. Follow with get_cached_energies and create_chart. Always specify model.",
        parameters={
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "enum": ["small", "medium", "large", "mace-mpa-0"],
                    "description": "small (fast), medium (balanced), large (accurate), mace-mpa-0 (best)",
                },
                "fmax": {
                    "type": "number",
                    "description": "Force threshold eV/A (default: 0.05)",
                },
                "maxSteps": {
                    "type": "integer",
                    "description": "Max steps (default: 100)",
                },
                "includeForces": {
                    "type": "boolean",
                    "description": "Include per-atom forces (default: true)",
                },
            },
        },
        required=["model"],
        execution_domain="server",
    ),
    ToolDefinition(
        name="run_md",
        description="MACE molecular dynamics (Langevin NVT). Returns trajectory frameCount and per-atom forces by default. Follow with get_cached_energies and create_chart. Always specify model. Use 'frames' to control exact output frame count (preferred over steps/saveInterval).",
        parameters={
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "enum": ["small", "medium", "large", "mace-mpa-0"],
                    "description": "MACE model",
                },
                "temperature": {
                    "type": "number",
                    "description": "Temp in K (default: 300)",
                },
                "frames": {
                    "type": "integer",
                    "description": "Exact number of output frames desired. Overrides steps/saveInterval. E.g. frames=10 produces exactly 10 frames.",
                },
                "steps": {
                    "type": "integer",
                    "description": "MD steps (default: 500). Ignored if frames is set.",
                },
                "timestep": {"type": "number", "description": "fs (default: 1.0)"},
                "friction": {"type": "number", "description": "1/fs (default: 0.01)"},
                "saveInterval": {
                    "type": "integer",
                    "description": "Save every N steps (default: 10). Ignored if frames is set.",
                },
                "includeForces": {
                    "type": "boolean",
                    "description": "Include per-atom forces (default: true)",
                },
            },
        },
        required=["model"],
        execution_domain="server",
    ),
    ToolDefinition(
        name="load_molecule",
        description="Load molecule by name from PubChem (e.g. caffeine, aspirin). Follow with get_molecule_info to inspect.",
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Molecule name"}},
        },
        required=["name"],
    ),
]

# ─── L5: Output tools ────────────────────────────────────────────────────────

OUTPUT_TOOLS = [
    ToolDefinition(
        name="create_chart",
        description="Display line/bar/scatter chart from x and y arrays. Use scanXValues from calculate_all_energies for x-axis. Style params let you customize colors, line width, point size, fill, grid, legend, etc.",
        parameters={
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["line", "bar", "scatter"],
                    "description": "Chart type (default: line)",
                },
                "title": {"type": "string", "description": "Chart title"},
                "xLabel": {"type": "string", "description": "X-axis label"},
                "yLabel": {"type": "string", "description": "Y-axis label"},
                "x": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "X values",
                },
                "y": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Y values",
                },
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Series labels",
                },
                "lineColor": {
                    "type": "string",
                    "description": "Line/border color as hex or CSS color (default: #667eea)",
                },
                "pointColor": {
                    "type": "string",
                    "description": "Point fill color (default: same as lineColor)",
                },
                "highlightColor": {
                    "type": "string",
                    "description": "Color for highlighted/current-frame point (default: #f093fb)",
                },
                "backgroundColor": {
                    "type": "string",
                    "description": "Chart area background color (default: rgba(0,0,0,0.3))",
                },
                "lineWidth": {
                    "type": "number",
                    "description": "Line thickness 1-6 (default: 2)",
                },
                "pointSize": {
                    "type": "number",
                    "description": "Point radius 0-10 (default: 3)",
                },
                "tension": {
                    "type": "number",
                    "description": "Curve smoothness 0-1 where 0=straight, 1=very smooth (default: 0.3)",
                },
                "fill": {
                    "type": "boolean",
                    "description": "Fill area under line (default: false)",
                },
                "fillColor": {
                    "type": "string",
                    "description": "Fill color with opacity e.g. rgba(102,126,234,0.2) (default: auto from lineColor)",
                },
                "showGrid": {
                    "type": "boolean",
                    "description": "Show grid lines (default: true)",
                },
                "showLegend": {
                    "type": "boolean",
                    "description": "Show chart legend (default: false)",
                },
                "showPoints": {
                    "type": "boolean",
                    "description": "Show data points (default: true)",
                },
                "fontSize": {
                    "type": "number",
                    "description": "Base font size for labels and ticks (default: 12)",
                },
                "datasets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "y": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "Y values for this series",
                            },
                            "label": {"type": "string", "description": "Series label"},
                            "color": {
                                "type": "string",
                                "description": "Line color for this series",
                            },
                        },
                        "required": ["y"],
                    },
                    "description": "Multiple data series (overrides y param). Each has y, label, color.",
                },
            },
        },
        required=["x", "y"],
        execution_domain="server",
    ),
    ToolDefinition(
        name="save_file",
        description="Export molecule to file (xyz, extxyz, mol, pdb, pqr, gro, mol2). Auto-includes forces/energies.",
        parameters={
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Filename (default: auto)",
                },
                "format": {
                    "type": "string",
                    "enum": ["xyz", "extxyz", "mol", "pdb", "pqr", "gro", "mol2"],
                    "description": "Format (default: xyz)",
                },
                "allFrames": {
                    "type": "boolean",
                    "description": "All frames (default: true)",
                },
                "saveToLocal": {
                    "type": "boolean",
                    "description": "Save to local folder vs download",
                },
            },
        },
    ),
    ToolDefinition(
        name="save_image",
        description="Save screenshot of current 3D view as PNG.",
        parameters={"type": "object", "properties": {}},
    ),
    ToolDefinition(
        name="create_file",
        description="Create a new file in the open folder.",
        parameters={
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Filename with extension",
                },
                "content": {"type": "string", "description": "File content"},
            },
        },
        required=["filename"],
    ),
    ToolDefinition(
        name="edit_file",
        description="Overwrite content of an AI-created file.",
        parameters={
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Filename to edit"},
                "content": {"type": "string", "description": "New content"},
            },
        },
        required=["filename", "content"],
    ),
    ToolDefinition(
        name="execute_python",
        description="Execute Python code. Pre-injected variables (use 'x' in dir() to check availability): atoms = list of dicts [{element, x, y, z}, ...] (current frame, Angstrom). positions = numpy float64 array shape (n_frames, n_atoms, 3) in Angstrom. energies = numpy float64 1D array of potential energies in eV. frames = list of dicts [{index, atoms}]. Libraries: numpy (np), matplotlib (plt), math. Figures auto-captured. Print results to stdout.",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
                "description": {
                    "type": "string",
                    "description": "Brief description of what this code does (shown to user for approval)",
                },
            },
        },
        required=["code"],
        execution_domain="server",
    ),
]

# ─── L6: View tools ──────────────────────────────────────────────────────────

VIEW_TOOLS = [
    ToolDefinition(
        name="toggle_labels",
        description="Show/hide atom labels. showElements for symbols (C, O), showIndices for numbers (0, 1), or both for combined (C0, O1).",
        parameters={
            "type": "object",
            "properties": {
                "showElements": {
                    "type": "boolean",
                    "description": "Show element symbols",
                },
                "showIndices": {"type": "boolean", "description": "Show atom indices"},
            },
        },
    ),
    ToolDefinition(
        name="toggle_force_arrows",
        description="Show/hide force vectors on atoms (green=low, red=high). Requires prior calculate_energy with includeForces:true.",
        parameters={"type": "object", "properties": {}},
    ),
    ToolDefinition(
        name="toggle_charge_visualization",
        description="Show/hide charge coloring and labels (red=negative, blue=positive). Requires ORCA charge data.",
        parameters={
            "type": "object",
            "properties": {
                "showColors": {
                    "type": "boolean",
                    "description": "Charge-based coloring",
                },
                "showLabels": {"type": "boolean", "description": "Charge value labels"},
                "chargeType": {
                    "type": "string",
                    "enum": ["mulliken", "loewdin"],
                    "description": "Charge type (default: mulliken)",
                },
            },
        },
    ),
    ToolDefinition(
        name="toggle_ribbon",
        description="Toggle protein backbone ribbon visualization.",
        parameters={"type": "object", "properties": {}},
    ),
    ToolDefinition(
        name="set_style",
        description="Change visual style: roughness, metalness, opacity, atomSize, backgroundColor.",
        parameters={
            "type": "object",
            "properties": {
                "roughness": {"type": "number", "description": "0-1"},
                "metalness": {"type": "number", "description": "0-1"},
                "opacity": {"type": "number", "description": "0-1 (1=solid)"},
                "atomSize": {"type": "number", "description": "0.1-3"},
                "backgroundColor": {"type": "string", "description": "Hex color"},
            },
        },
    ),
    ToolDefinition(
        name="show_all_bond_lengths",
        description="Display bond length labels on every bond.",
        parameters={"type": "object", "properties": {}},
    ),
    ToolDefinition(
        name="remove_bond_label",
        description="Remove specific bond label (atom1+atom2) or all labels (all:true).",
        parameters={
            "type": "object",
            "properties": {
                "atom1": {"type": "integer", "description": "First atom"},
                "atom2": {"type": "integer", "description": "Second atom"},
                "all": {"type": "boolean", "description": "Remove all labels"},
            },
        },
    ),
    ToolDefinition(
        name="clear_measurements",
        description="Remove all distance/angle/dihedral measurement labels.",
        parameters={"type": "object", "properties": {}},
    ),
    ToolDefinition(
        name="reset_camera",
        description="Reset camera to default view.",
        parameters={"type": "object", "properties": {}},
    ),
    ToolDefinition(
        name="zoom_to_fit",
        description="Zoom camera to fit entire molecule.",
        parameters={"type": "object", "properties": {}},
    ),
    ToolDefinition(
        name="rotate_camera",
        description="Rotate camera view by angle degrees.",
        parameters={
            "type": "object",
            "properties": {"angle": {"type": "number", "description": "Degrees"}},
        },
        required=["angle"],
    ),
    ToolDefinition(
        name="define_axis",
        description="Define rotation/translation axis from 2 selected atoms. Required before manual transform_atoms.",
        parameters={"type": "object", "properties": {}},
    ),
    ToolDefinition(
        name="remove_axis",
        description="Remove the defined axis.",
        parameters={"type": "object", "properties": {}},
    ),
    ToolDefinition(
        name="create_fragment",
        description="Group selected atoms into a named fragment.",
        parameters={"type": "object", "properties": {}},
    ),
    ToolDefinition(
        name="isolate_selection",
        description="Isolate selected atoms to view/edit separately.",
        parameters={"type": "object", "properties": {}},
    ),
    ToolDefinition(
        name="undo",
        description="Undo last action.",
        parameters={"type": "object", "properties": {}},
    ),
    ToolDefinition(
        name="redo",
        description="Redo last undone action.",
        parameters={"type": "object", "properties": {}},
    ),
]


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for group in [
        QUERY_TOOLS,
        SELECTION_TOOLS,
        EDIT_TOOLS,
        GENERATION_TOOLS,
        OUTPUT_TOOLS,
        VIEW_TOOLS,
    ]:
        registry.register_many(group)
    return registry
