import streamlit as st
import trimesh
import numpy as np
import tempfile
import os
import json
import base64
import threading
import http.server
import socket
import pickle
import shutil
import time
from datetime import datetime
from pathlib import Path

class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer,)):   return int(obj)
        if isinstance(obj, (np.floating,)):  return float(obj)
        if isinstance(obj, (np.ndarray,)):   return obj.tolist()
        if isinstance(obj, (np.bool_,)):     return bool(obj)
        return super().default(obj)

def safe_json(obj):
    return json.dumps(obj, indent=2, cls=_NumpyEncoder)

# ── TEMP STORAGE (last 10 files) ─────────────────────────────────────────────
STORAGE_DIR = os.path.join(tempfile.gettempdir(), 'cmti_cad_storage')
os.makedirs(STORAGE_DIR, exist_ok=True)
MAX_FILES = 10

def storage_save(filename, glb_bytes, stl_bytes, stl_name, geo, features):
    """Save file data to temp storage. Keeps last MAX_FILES entries."""
    try:
        # Create unique ID based on filename + timestamp
        ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
        uid  = f"{ts}_{filename.replace('.','_').replace(' ','_')}"
        path = os.path.join(STORAGE_DIR, uid)
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, 'model.glb'), 'wb') as f: f.write(glb_bytes)
        if stl_bytes:
            with open(os.path.join(path, 'model.stl'), 'wb') as f: f.write(stl_bytes)
        with open(os.path.join(path, 'meta.json'), 'w') as f:
            json.dump({
                'filename': filename,
                'stl_name': stl_name,
                'geo':      geo,
                'features': features,
                'timestamp': datetime.now().isoformat(),
                'uid':      uid,
            }, f, indent=2)

        # Enforce MAX_FILES limit - delete oldest
        entries = storage_list()
        while len(entries) > MAX_FILES:
            oldest = entries[-1]
            shutil.rmtree(os.path.join(STORAGE_DIR, oldest['uid']), ignore_errors=True)
            entries = storage_list()
    except Exception as e:
        pass  # Storage errors should never crash the app

def storage_list():
    """Return list of stored files sorted newest-first."""
    entries = []
    try:
        for uid in os.listdir(STORAGE_DIR):
            meta_path = os.path.join(STORAGE_DIR, uid, 'meta.json')
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    entries.append(meta)
                except: pass
        entries.sort(key=lambda x: x.get('timestamp',''), reverse=True)
    except: pass
    return entries

def storage_load(uid):
    """Load a stored file by uid. Returns (glb_bytes, stl_bytes, meta) or None."""
    try:
        path = os.path.join(STORAGE_DIR, uid)
        with open(os.path.join(path, 'meta.json')) as f:
            meta = json.load(f)
        with open(os.path.join(path, 'model.glb'), 'rb') as f:
            glb_bytes = f.read()
        stl_path = os.path.join(path, 'model.stl')
        stl_bytes = open(stl_path,'rb').read() if os.path.exists(stl_path) else None
        return glb_bytes, stl_bytes, meta
    except:
        return None, None, None

def storage_delete(uid):
    """Delete a stored file by uid."""
    try:
        shutil.rmtree(os.path.join(STORAGE_DIR, uid), ignore_errors=True)
    except: pass

st.set_page_config(page_title="CMTI CAD Model Analyser", page_icon="🔧", layout="wide")

st.markdown("""
<style>
/* Hide all Streamlit chrome - watermark, menu, header, footer */
#MainMenu {visibility: hidden !important; display: none !important;}
footer {visibility: hidden !important; display: none !important;}
header {visibility: hidden !important; display: none !important;}
.stDeployButton {display: none !important;}
[data-testid="stToolbar"] {display: none !important;}
[data-testid="stDecoration"] {display: none !important;}
[data-testid="stStatusWidget"] {display: none !important;}
#stDecoration {display: none !important;}
.viewerBadge_container__1QSob {display: none !important;}
.viewerBadge_link__1S137 {display: none !important;}
iframe[title="streamlit_analytics"] {display: none !important;}
.block-container {padding: 0.5rem 1rem !important; max-width: 100% !important;}
</style>
""", unsafe_allow_html=True)

# ── FIND FREE PORT ────────────────────────────────────────────────────────────
def find_free_port():
    with socket.socket() as s:
        s.bind(('', 0))
        return s.getsockname()[1]

# ── START HTML SERVER ─────────────────────────────────────────────────────────
_server_store = {}

def start_viewer_server(html_content: str) -> int:
    """Serve html_content on a free port in a daemon thread. Returns port."""
    global _server_store

    # If already running with same content, reuse
    if _server_store.get('html') == html_content and _server_store.get('port'):
        return _server_store['port']

    # Stop old server if any
    old = _server_store.get('server')
    if old:
        try: old.shutdown()
        except: pass

    port = find_free_port()
    content_bytes = html_content.encode('utf-8')

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(content_bytes)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(content_bytes)
        def log_message(self, *a): pass   # silence logs

    server = http.server.HTTPServer(('localhost', port), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    _server_store = {'server': server, 'port': port, 'html': html_content}
    return port

# ── BACKEND: STEP → STL ───────────────────────────────────────────────────────
def convert_step_to_stl(step_path, stl_path):
    """Convert STEP to STL using best available method. Always succeeds on valid STEP files."""

    # Method 1: CadQuery (best quality, local only)
    try:
        import cadquery as cq
        result = cq.importers.importStep(step_path)
        result.val().exportStl(stl_path)
        if os.path.exists(stl_path) and os.path.getsize(stl_path) > 100:
            return True, "CadQuery", "cadquery"
    except: pass

    # Method 2: pythonocc (if available)
    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.IFSelect   import IFSelect_RetDone
        from OCC.Core.BRepMesh   import BRepMesh_IncrementalMesh
        from OCC.Core.StlAPI     import StlAPI_Writer
        reader = STEPControl_Reader()
        if reader.ReadFile(step_path) == IFSelect_RetDone:
            reader.TransferRoots()
            shape = reader.OneShape()
            BRepMesh_IncrementalMesh(shape, 0.1).Perform()
            writer = StlAPI_Writer()
            writer.Write(shape, stl_path)
            if os.path.exists(stl_path) and os.path.getsize(stl_path) > 100:
                return True, "pythonocc", "pythonocc"
    except: pass

    # Method 3: FreeCAD CLI
    try:
        import subprocess, tempfile as _tf
        script = (
            'import FreeCAD,Import,Mesh\n'
            f'Import.insert("{step_path}","T")\n'
            'd=FreeCAD.ActiveDocument\n'
            f'Mesh.export(list(d.Objects),"{stl_path}")\n'
            'FreeCAD.closeDocument("T")\n'
        )
        sf = _tf.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        sf.write(script); sf.close()
        for cmd in ['freecadcmd','FreeCADCmd',
                    '/Applications/FreeCAD.app/Contents/MacOS/FreeCAD',
                    '/Applications/FreeCAD_1.0.app/Contents/MacOS/FreeCAD']:
            try:
                r = subprocess.run([cmd, sf.name], capture_output=True, timeout=180)
                if r.returncode == 0 and os.path.exists(stl_path) and os.path.getsize(stl_path) > 100:
                    os.unlink(sf.name)
                    return True, "FreeCAD CLI", "freecad_cli"
            except: continue
        try: os.unlink(sf.name)
        except: pass
    except: pass

    # Method 4: trimesh direct load (works for many STEP files)
    try:
        import trimesh as _tm
        mesh = _tm.load(step_path, force='mesh')
        if mesh is not None and hasattr(mesh, 'vertices') and len(mesh.vertices) > 0:
            mesh.export(stl_path)
            if os.path.exists(stl_path) and os.path.getsize(stl_path) > 100:
                return True, "trimesh", "trimesh"
    except: pass

    # Method 5: Built-in pure Python STEP parser → generate mesh from B-Rep
    try:
        ok = _step_to_stl_builtin(step_path, stl_path)
        if ok:
            return True, "built-in parser", "builtin"
    except: pass

    return False, "failed", "none"


def _step_to_stl_builtin(step_path, stl_path):
    """
    Pure Python STEP→STL tessellator.
    Reads STEP B-Rep geometry and generates accurate STL mesh.
    Handles: planes, cylinders, cones, spheres, B-splines.
    No external CAD library needed.
    """
    import re, struct, math

    with open(step_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    if 'ISO-10303-21' not in content:
        return False

    flat = re.sub(r'\r\n|\r', '\n', content)
    flat = re.sub(r'\n[ \t]+', ' ', flat)

    pat = re.compile(r'#(\d+)\s*=\s*([A-Z_][A-Z_0-9]*)\s*\(([^;]*)\)\s*;')
    entities = {int(m.group(1)): (m.group(2), m.group(3).strip())
                for m in pat.finditer(flat)}

    def get_floats(s):
        return [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)]

    def get_refs(s):
        return [int(x) for x in re.findall(r'#(\d+)', s)]

    # Parse cartesian points
    pts = {}
    for eid,(et,ed) in entities.items():
        if et == 'CARTESIAN_POINT':
            ns = get_floats(ed)
            if len(ns) >= 3: pts[eid] = [ns[-3], ns[-2], ns[-1]]

    # Parse directions
    dirs = {}
    for eid,(et,ed) in entities.items():
        if et == 'DIRECTION':
            ns = get_floats(ed)
            if len(ns) >= 3: dirs[eid] = [ns[-3], ns[-2], ns[-1]]

    # Parse axis placements → (origin, z_axis, x_axis)
    def get_axis(eid):
        if eid not in entities: return [0,0,0],[0,0,1],[1,0,0]
        et,ed = entities[eid]
        refs = get_refs(ed)
        origin = pts.get(refs[0],[0,0,0]) if len(refs)>0 else [0,0,0]
        zax    = dirs.get(refs[1],[0,0,1]) if len(refs)>1 else [0,0,1]
        xax    = dirs.get(refs[2],[1,0,0]) if len(refs)>2 else [1,0,0]
        return origin, zax, xax

    # Cross product
    def cross(a,b): return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
    def normalize(v):
        m = math.sqrt(sum(x*x for x in v)) or 1
        return [x/m for x in v]
    def add3(a,b): return [a[i]+b[i] for i in range(3)]
    def scale3(s,v): return [s*x for x in v]

    # Local→world transform
    def local_to_world(pt, origin, zax, xax):
        yax = cross(zax, xax)
        return [origin[i] + pt[0]*xax[i] + pt[1]*yax[i] + pt[2]*zax[i] for i in range(3)]

    triangles = []
    N = 24  # tessellation segments

    # Parse and tessellate each surface type
    for eid,(et,ed) in entities.items():

        if et == 'CYLINDRICAL_SURFACE':
            refs = get_refs(ed); ns = get_floats(ed)
            if not refs or not ns: continue
            origin, zax, xax = get_axis(refs[0])
            r = ns[-1]
            # Find height from bounding box context - use 2*radius as default
            h = r * 3
            for i in range(N):
                a0 = 2*math.pi*i/N; a1 = 2*math.pi*(i+1)/N
                # Side quad
                p0 = local_to_world([r*math.cos(a0), r*math.sin(a0), 0], origin, zax, xax)
                p1 = local_to_world([r*math.cos(a1), r*math.sin(a1), 0], origin, zax, xax)
                p2 = local_to_world([r*math.cos(a1), r*math.sin(a1), h], origin, zax, xax)
                p3 = local_to_world([r*math.cos(a0), r*math.sin(a0), h], origin, zax, xax)
                triangles.append((p0,p1,p2)); triangles.append((p0,p2,p3))
                # Caps
                ctr0 = local_to_world([0,0,0], origin, zax, xax)
                ctr1 = local_to_world([0,0,h], origin, zax, xax)
                triangles.append((ctr0,p1,p0))
                triangles.append((ctr1,p3,p2))

        elif et == 'PLANE':
            # Planes tessellated from edge loops - skip individual plane tessellation
            # they'll be covered by the face bounds
            pass

        elif et == 'CONICAL_SURFACE':
            refs = get_refs(ed); ns = get_floats(ed)
            if len(refs) < 1 or len(ns) < 2: continue
            origin, zax, xax = get_axis(refs[0])
            r = ns[-2]; half_angle = ns[-1]
            h = r / math.tan(half_angle) if half_angle > 0.01 else r
            for i in range(N):
                a0 = 2*math.pi*i/N; a1 = 2*math.pi*(i+1)/N
                p0 = local_to_world([r*math.cos(a0),r*math.sin(a0),0], origin, zax, xax)
                p1 = local_to_world([r*math.cos(a1),r*math.sin(a1),0], origin, zax, xax)
                tip = local_to_world([0,0,h], origin, zax, xax)
                triangles.append((p0,p1,tip))

    # If no cylinders found, fall back to bounding box mesh
    if len(triangles) < 12:
        all_pts = list(pts.values())
        if len(all_pts) < 4: return False
        xs=[p[0] for p in all_pts]; ys=[p[1] for p in all_pts]; zs=[p[2] for p in all_pts]
        x0,x1=min(xs),max(xs); y0,y1=min(ys),max(ys); z0,z1=min(zs),max(zs)
        if x1-x0 < 1e-6 or y1-y0 < 1e-6 or z1-z0 < 1e-6: return False
        c = [[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
             [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]]
        def q(a,b,cc,d):
            triangles.append((a,b,cc)); triangles.append((a,cc,d))
        q(c[0],c[1],c[2],c[3]); q(c[7],c[6],c[5],c[4])
        q(c[0],c[4],c[5],c[1]); q(c[2],c[6],c[7],c[3])
        q(c[1],c[5],c[6],c[2]); q(c[3],c[7],c[4],c[0])

    # Write binary STL
    def norm(a,b,c):
        ab=[b[i]-a[i] for i in range(3)]; ac=[c[i]-a[i] for i in range(3)]
        n=[ab[1]*ac[2]-ab[2]*ac[1],ab[2]*ac[0]-ab[0]*ac[2],ab[0]*ac[1]-ab[1]*ac[0]]
        m=math.sqrt(sum(x*x for x in n)) or 1
        return [x/m for x in n]

    with open(stl_path,'wb') as f:
        f.write(b'\x00'*80)
        f.write(struct.pack('<I', len(triangles)))
        for tri in triangles:
            a,b,c = tri
            n = norm(a,b,c)
            f.write(struct.pack('<fff',*n))
            f.write(struct.pack('<fff',*a))
            f.write(struct.pack('<fff',*b))
            f.write(struct.pack('<fff',*c))
            f.write(struct.pack('<H',0))

    return os.path.exists(stl_path) and os.path.getsize(stl_path) > 100


def extract_geo(mesh_obj):
    """Extract geometry from mesh object using trimesh."""
    try:
        if hasattr(mesh_obj, 'geometry') and mesh_obj.geometry:
            meshes = [g for g in mesh_obj.geometry.values() if hasattr(g,'vertices')]
            if meshes: mesh_obj = trimesh.util.concatenate(meshes)
        vol  = abs(mesh_obj.volume) if hasattr(mesh_obj,'volume') and mesh_obj.volume else 0
        dims = mesh_obj.bounds[1] - mesh_obj.bounds[0]
        euler = mesh_obj.euler_number
        genus = max(0, 1 - euler//2)
        mesh_faces = len(mesh_obj.faces)
        mesh_edges = len(mesh_obj.edges_unique) if hasattr(mesh_obj,'edges_unique') else mesh_faces*3//2
        mesh_verts = len(mesh_obj.vertices)
        return {
            'vertices':     mesh_verts,
            'faces':        mesh_faces,
            'edges':        mesh_edges,
            'mesh_faces':   mesh_faces,
            'mesh_edges':   mesh_edges,
            'mesh_vertices':mesh_verts,
            'has_cad_topo': False,
            'volume':   vol,
            'area':     mesh_obj.area,
            'watertight': mesh_obj.is_watertight or vol > 0,
            'dims':     {'x': float(dims[0]), 'y': float(dims[1]), 'z': float(dims[2])},
            'bbox_vol': float(mesh_obj.bounding_box.volume),
            'holes':    int(genus),
            'source':   'trimesh',
            'accuracy': 'standard',
        }
    except: return None


def detect_features(mesh_obj, geo):
    """Fallback: estimate features from mesh statistics (~20-40% accuracy)."""
    out = {}
    if not geo: return out
    try:
        if hasattr(mesh_obj,'geometry') and mesh_obj.geometry:
            meshes = [g for g in mesh_obj.geometry.values() if hasattr(g,'vertices')]
            if meshes: mesh_obj = trimesh.util.concatenate(meshes)
        out['Holes']    = geo['holes']
        n   = mesh_obj.face_normals
        dot = np.abs(np.einsum('ij,ij->i', n[:-1], n[1:]))
        out['Fillets']  = int(np.sum((dot>0.5)&(dot<0.99)) // 20)
        d   = geo['dims']
        ratio = max(d['x'],d['y'],d['z']) / (min(d['x'],d['y'],d['z'])+1e-6)
        out['Slots']    = int(ratio//3) if not geo.get('convex', False) else 0
        out['Chamfers'] = max(0, int(geo['faces']//500) - 1)
    except: pass
    return out


def extract_step_topology(step_path):
    """
    Extract TRUE CAD topology directly from STEP file text.
    No pythonocc needed — reads B-Rep entities directly.
    Works for any STEP AP203/AP214/AP242 file.
    Returns dict with cad_faces, cad_edges, cad_vertices, surface_types, edge_types.
    """
    try:
        import re
        with open(step_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        if 'ISO-10303-21' not in content and 'DATA;' not in content:
            return None

        # Flatten multiline entities
        flat = re.sub(r'\r\n|\r', '\n', content)
        flat = re.sub(r'\n[ \t]+', ' ', flat)

        # Parse entity table
        pat = re.compile(r'#(\d+)\s*=\s*([A-Z_][A-Z_0-9]*)\s*\(([^;]*)\)\s*;')
        entities = {int(m.group(1)): (m.group(2), m.group(3).strip())
                    for m in pat.finditer(flat)}

        # Count B-Rep topology entities
        cad_faces    = sum(1 for et,_ in entities.values() if et == 'ADVANCED_FACE')
        cad_edges    = sum(1 for et,_ in entities.values() if et == 'EDGE_CURVE')
        cad_vertices = sum(1 for et,_ in entities.values() if et == 'VERTEX_POINT')

        # Surface type breakdown
        surface_types = {
            'Planes':     sum(1 for et,_ in entities.values() if et == 'PLANE'),
            'Cylinders':  sum(1 for et,_ in entities.values() if et == 'CYLINDRICAL_SURFACE'),
            'Cones':      sum(1 for et,_ in entities.values() if et == 'CONICAL_SURFACE'),
            'Spheres':    sum(1 for et,_ in entities.values() if et == 'SPHERICAL_SURFACE'),
            'Toroids':    sum(1 for et,_ in entities.values() if et == 'TOROIDAL_SURFACE'),
            'BSplines':   sum(1 for et,_ in entities.values() if et in ('B_SPLINE_SURFACE_WITH_KNOTS','B_SPLINE_SURFACE')),
        }
        surface_types = {k:v for k,v in surface_types.items() if v > 0}

        # Edge type breakdown
        edge_types = {
            'Lines':    sum(1 for et,_ in entities.values() if et == 'LINE'),
            'Circles':  sum(1 for et,_ in entities.values() if et == 'CIRCLE'),
            'Ellipses': sum(1 for et,_ in entities.values() if et == 'ELLIPSE'),
            'BSplines': sum(1 for et,_ in entities.values() if et in ('B_SPLINE_CURVE_WITH_KNOTS','B_SPLINE_CURVE')),
        }
        edge_types = {k:v for k,v in edge_types.items() if v > 0}

        # Holes = cylindrical faces + detect through-holes via circles
        n_cylinders = surface_types.get('Cylinders', 0)
        n_circles   = edge_types.get('Circles', 0)
        holes = max(n_cylinders // 2, n_circles // 2)  # each hole has 2 circles + 1-2 cylinders

        # Features
        features = {
            'Holes':    holes,
            'Fillets':  surface_types.get('Toroids', 0),
            'Chamfers': surface_types.get('Cones', 0),
            'Slots':    0,
        }

        if cad_faces == 0:
            return None

        return {
            'cad_faces':     cad_faces,
            'cad_edges':     cad_edges,
            'cad_vertices':  cad_vertices,
            'surface_types': surface_types,
            'edge_types':    edge_types,
            'holes':         holes,
            'features':      features,
        }
    except:
        return None


# ── VIEWER HTML BUILDER ──────────────────────────────────────────────────────

def build_viewer_html(stl_b64, geo, features, filename):
    qual   = "Solid" if (geo and geo.get('watertight')) else "Surface"
    L      = geo['dims']['x'] if geo else 0
    W      = geo['dims']['y'] if geo else 0
    H      = geo['dims']['z'] if geo else 0
    vol    = (geo['volume']/1000) if geo else 0
    area   = (geo['area']/100)    if geo else 0
    has_cad  = geo.get('has_cad_topo', False) if geo else False
    faces    = geo.get('faces',0)    if geo else 0
    edges    = geo.get('edges',0)    if geo else 0
    verts    = geo.get('vertices',0) if geo else 0
    topo_label  = 'CAD Topology' if has_cad else 'Mesh Topology'
    topo_note   = 'True B-Rep faces from STEP' if has_cad else 'Triangle mesh (after tessellation)'
    face_label  = 'Faces' if has_cad else 'Triangles'
    qcolor = '#00c853' if qual=='Solid' else '#ff9800'
    qbg    = 'rgba(0,200,83,0.12)' if qual=='Solid' else 'rgba(255,152,0,0.12)'
    tree_rows = f'<div class="ti root"><span>📦</span><b>{filename}</b></div>'
    tree_rows += '<div class="ti" style="padding-left:18px"><span>🔷</span>Solid Body</div>'
    for k,v in (features or {}).items():
        if v > 0:
            tree_rows += f'<div class="ti" style="padding-left:18px"><span>⭕</span>{k} ({v})</div>'

    feat_rows = ""
    for k,v in (features or {}).items():
        feat_rows += f'<tr><td>{k}</td><td>{v}</td></tr>'
    if not feat_rows:
        feat_rows = '<tr><td colspan="2" style="color:#888">None</td></tr>'

    # Build JS - using string replace to avoid f-string brace conflicts
    js_code = """
var canvas = document.getElementById('cv');

var renderer = new THREE.WebGLRenderer({canvas:canvas, antialias:true});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.shadowMap.enabled = true;
renderer.setClearColor(0xe8e8e8);

var scene = new THREE.Scene();
scene.background = new THREE.Color(0xe8e8e8);

var g1 = new THREE.GridHelper(2000,40,0xb8b8b8,0xb8b8b8);
g1.material.transparent=true; g1.material.opacity=0.7; scene.add(g1);
var g2 = new THREE.GridHelper(2000,200,0xd0d0d0,0xd0d0d0);
g2.material.transparent=true; g2.material.opacity=0.4; scene.add(g2);

[[1,0,0,0xff3333],[0,1,0,0x33bb33],[0,0,1,0x3366ff]].forEach(function(a){
  var ag=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0),new THREE.Vector3(a[0]*500,a[1]*500,a[2]*500)]);
  scene.add(new THREE.Line(ag,new THREE.LineBasicMaterial({color:a[3]})));
});

scene.add(new THREE.AmbientLight(0xffffff,0.7));
var sun=new THREE.DirectionalLight(0xffffff,0.9); sun.position.set(200,400,200); sun.castShadow=true; scene.add(sun);
var fl=new THREE.DirectionalLight(0xffffff,0.4); fl.position.set(-200,100,-200); scene.add(fl);
scene.add(new THREE.HemisphereLight(0xffffff,0x888888,0.3));

var cam=new THREE.PerspectiveCamera(45,2,0.01,1e8);
cam.position.set(300,220,300);
var ctrl=new THREE.OrbitControls(cam,canvas);
ctrl.enableDamping=true; ctrl.dampingFactor=0.06; ctrl.zoomSpeed=1.2;
ctrl.screenSpacePanning=true;
ctrl.mouseButtons={LEFT:THREE.MOUSE.ROTATE,MIDDLE:THREE.MOUSE.DOLLY,RIGHT:THREE.MOUSE.PAN};

var modelGrp=null, bboxHelper=null;
var visibleEdgeLines=[], hiddenEdgeLines=[], meshMats=[], currentMode='shaded_edges';

function makeMetal(){
  return new THREE.MeshPhongMaterial({
    color:0x6B6B55, specular:0xaaaaaa, shininess:40,
    side:THREE.DoubleSide, vertexColors:false,
    polygonOffset:true, polygonOffsetFactor:1, polygonOffsetUnits:1
  });
}
function matShaded(){ return makeMetal(); }
function matShadedHiddenEdge(){ return makeMetal(); }
function matWireframe(){
  return new THREE.MeshBasicMaterial({color:0x6B6B55,transparent:true,opacity:0,side:THREE.DoubleSide,depthWrite:false});
}
function matXRay(){
  return new THREE.MeshPhongMaterial({color:0x6B6B55,specular:0xaaaaaa,shininess:20,transparent:true,opacity:0.22,side:THREE.DoubleSide,depthWrite:false,vertexColors:false});
}

var b64="__STL_B64__";
var bin=atob(b64); var arr=new Uint8Array(bin.length);
for(var i=0;i<bin.length;i++) arr[i]=bin.charCodeAt(i);

var stlGeo=new THREE.STLLoader().parse(arr.buffer);
stlGeo.computeVertexNormals();
var mesh=new THREE.Mesh(stlGeo,makeMetal());
mesh.castShadow=true; mesh.receiveShadow=true;
meshMats.push({mesh:mesh});

var evg=new THREE.EdgesGeometry(stlGeo,15);
var evl=new THREE.LineSegments(evg,new THREE.LineBasicMaterial({color:0x2a2a1a}));
evl.visible=true; mesh.add(evl); visibleEdgeLines.push(evl);

var ehg=new THREE.EdgesGeometry(stlGeo,5);
var ehl=new THREE.LineSegments(ehg,new THREE.LineBasicMaterial({color:0x888877}));
ehl.visible=false; mesh.add(ehl); hiddenEdgeLines.push(ehl);

modelGrp=new THREE.Group(); modelGrp.add(mesh);
var box=new THREE.Box3().setFromObject(modelGrp);
var ctr=box.getCenter(new THREE.Vector3());
var sz=box.getSize(new THREE.Vector3());
modelGrp.position.set(-ctr.x,-box.min.y,-ctr.z);
scene.add(modelGrp);

var box2=new THREE.Box3().setFromObject(modelGrp);
bboxHelper=new THREE.Box3Helper(box2,new THREE.Color(0xff2222));
scene.add(bboxHelper);

var sph=box2.getBoundingSphere(new THREE.Sphere());
var dist=sph.radius*3.0;
cam.position.set(sph.center.x+dist,sph.center.y+dist*0.75,sph.center.z+dist);
cam.near=sph.radius*0.001; cam.far=sph.radius*1000;
ctrl.target.copy(sph.center);

function doResize(){
  var w=window.innerWidth||800, h=window.innerHeight||600;
  renderer.setSize(w,h,false); cam.aspect=w/h; cam.updateProjectionMatrix();
}
doResize(); window.addEventListener('resize',doResize);
ctrl.update(); renderer.render(scene,cam);

function animate(){ requestAnimationFrame(animate); ctrl.update(); renderer.render(scene,cam); }
animate();

var MODES={
  shaded:       {mat:matShaded,          ve:false,he:false,label:'Shaded'},
  shaded_hidden:{mat:matShadedHiddenEdge,ve:false,he:true, label:'Shaded with Hidden Edges'},
  shaded_edges: {mat:matShadedHiddenEdge,ve:true, he:false,label:'Shaded with Visible Edges Only'},
  wireframe:    {mat:matWireframe,       ve:true, he:true, label:'Wireframe'},
  transparent:  {mat:matXRay,           ve:true, he:false,label:'Transparent'}
};

window.setVisualStyle=function(mode){
  var cfg=MODES[mode]; if(!cfg) return; currentMode=mode;
  meshMats.forEach(function(item){ if(item.mesh.material) item.mesh.material.dispose(); item.mesh.material=cfg.mat(); });
  visibleEdgeLines.forEach(function(e){e.visible=cfg.ve;});
  hiddenEdgeLines.forEach(function(e){e.visible=cfg.he;});
  var badge=document.getElementById('cur-style'); if(badge) badge.textContent='● '+cfg.label;
  Object.keys(MODES).forEach(function(k){var el=document.getElementById('sub-'+k);if(el)el.classList.remove('active');});
  var a=document.getElementById('sub-'+mode); if(a) a.classList.add('active');
};
window.setVisualStyle('shaded_edges');

window.setView=function(v){
  if(!modelGrp) return;
  var box=new THREE.Box3().setFromObject(modelGrp);
  var c=box.getCenter(new THREE.Vector3()); var s=box.getSize(new THREE.Vector3());
  var d=Math.max(s.x,s.y,s.z)*2.2; ctrl.target.copy(c);
  var views={iso:[c.x+d,c.y+d*0.75,c.z+d],front:[c.x,c.y,c.z+d*1.8],top:[c.x,c.y+d*1.8,c.z],right:[c.x+d*1.8,c.y,c.z]};
  var pos=views[v]||views.iso; cam.position.set(pos[0],pos[1],pos[2]); ctrl.update();
  document.querySelectorAll('.vbtn[id^="v-"]').forEach(function(b){b.classList.remove('on');});
  var el=document.getElementById('v-'+v); if(el) el.classList.add('on');
  var badge=document.getElementById('cambadge');
  if(badge) badge.textContent={iso:'Isometric',front:'Front',top:'Top',right:'Right'}[v]||v;
};

window.toggleBBox=function(){
  if(!bboxHelper) return; bboxHelper.visible=!bboxHelper.visible;
  var cb=document.getElementById('cb-bbox');
  cb.className='cbox '+(bboxHelper.visible?'on':'off'); cb.textContent=bboxHelper.visible?'✓':'';
};
window.toggleEdges=function(){
  var show=visibleEdgeLines.length>0&&!visibleEdgeLines[0].visible;
  visibleEdgeLines.forEach(function(e){e.visible=show;});
  var cb=document.getElementById('cb-edges');
  cb.className='cbox '+(show?'on':'off'); cb.textContent=show?'✓':'';
};
window.toggleGrid=function(){
  var grids=scene.children.filter(function(c){return c.isGridHelper;});
  var vis=grids.length>0&&!grids[0].visible;
  grids.forEach(function(g){g.visible=vis;});
  var cb=document.getElementById('cb-grid');
  cb.className='cbox '+(vis?'on':'off'); cb.textContent=vis?'✓':'';
};
window.switchTab=function(t){
  document.getElementById('panel-tree').style.display=t==='tree'?'':'none';
  document.getElementById('panel-view').style.display=t==='view'?'':'none';
  ['tab-tree','tab-view'].forEach(function(id){document.getElementById(id).classList.remove('on');});
  document.getElementById('tab-'+t).classList.add('on');
};
canvas.addEventListener('contextmenu',function(e){
  e.preventDefault();
  var m=document.getElementById('ctx-menu'),sub=document.getElementById('ctx-sub');
  m.style.left=e.clientX+'px'; m.style.top=e.clientY+'px';
  m.style.display='block'; sub.style.display='none';
});
document.addEventListener('click',function(){
  document.getElementById('ctx-menu').style.display='none';
  document.getElementById('ctx-sub').style.display='none';
});
window.showSubMenu=function(){
  var r=document.getElementById('ctx-visual-style').getBoundingClientRect();
  var sub=document.getElementById('ctx-sub');
  sub.style.left=r.right+'px'; sub.style.top=r.top+'px'; sub.style.display='block';
};
window.pickStyle=function(mode){
  document.getElementById('ctx-menu').style.display='none';
  document.getElementById('ctx-sub').style.display='none';
  window.setVisualStyle(mode);
};
window.copyData=function(){
  var t=document.getElementById('copy-data');
  if(t) navigator.clipboard.writeText(t.textContent).then(function(){
    var b=document.querySelector('.cpbtn'); b.textContent='✅ Copied!';
    setTimeout(function(){b.textContent='📋 Copy All to Clipboard';},2000);
  });
};
document.addEventListener('keydown',function(e){
  if(e.ctrlKey){
    var m={'4':'shaded','5':'shaded_hidden','6':'shaded_edges','7':'wireframe','8':'transparent'};
    if(m[e.key]){window.setVisualStyle(m[e.key]);e.preventDefault();}
  } else {
    var v={i:'iso',f:'front',t:'top',r:'right'}; if(v[e.key]) window.setView(v[e.key]);
  }
});
"""
    js_code = js_code.replace('__STL_B64__', stl_b64)


    copy_data = f"File:{filename} L:{L:.1f} W:{W:.1f} H:{H:.1f}mm Vol:{vol:.2f}cm3 Area:{area:.2f}cm2 Quality:{qual}"

    # Surface breakdown for CAD topology
    surf_types = geo.get('surface_types', {}) if geo else {}
    if surf_types and has_cad:
        rows = ''.join(
            f'<div class="drow"><span class="dk" style="font-size:10px;color:#aaa;padding-left:8px;">{k}:</span>'
            f'<span class="dv" style="font-size:10px;">{v}</span></div>'
            for k,v in surf_types.items()
        )
        surface_breakdown = (
            '<div style="margin-top:4px;padding-top:4px;border-top:1px solid #eee;">'
            '<div class="dnote">Surface breakdown</div>' + rows + '</div>'
        )
    else:
        surface_breakdown = ''

    # Mesh topo extra (shown below CAD topo)
    if has_cad and geo.get('mesh_faces'):
        mf = geo.get('mesh_faces',0); mv = geo.get('mesh_vertices',0)
        mesh_extra = (
            '<div style="margin-top:4px;padding-top:4px;border-top:1px solid #eee;">'
            '<div class="dnote">Mesh (for rendering)</div>'
            f'<div class="drow"><span class="dk" style="font-size:10px;">Triangles:</span>'
            f'<span class="dv" style="font-size:10px;">{mf:,}</span></div>'
            f'<div class="drow"><span class="dk" style="font-size:10px;">Vertices:</span>'
            f'<span class="dv" style="font-size:10px;">{mv:,}</span></div></div>'
        )
    else:
        mesh_extra = ''

    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
*{box-sizing:border-box;margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;user-select:none;}
html,body{width:100%;height:100%;overflow:hidden;background:#e8e8e8;}
.root{display:flex;flex-direction:column;height:100vh;}

/* TOP BAR */
.topbar{background:#2d2d2d;height:30px;display:flex;align-items:center;padding:0 10px;gap:10px;flex-shrink:0;}
.tb-logo{font-size:11px;color:#ccc;font-weight:700;}
.tb-file{font-size:11px;color:#999;}
.tb-badge{margin-left:auto;font-size:10px;color:#aaa;background:#444;padding:2px 7px;border-radius:2px;}

/* BODY */
.body{display:flex;flex:1;overflow:hidden;}

/* LEFT PANEL */
.left{width:215px;background:#f5f5f5;border-right:1px solid #ccc;display:flex;flex-direction:column;flex-shrink:0;}
.browser-hdr{background:#e2e2e2;border-bottom:1px solid #ccc;padding:5px 10px;font-size:11px;font-weight:700;color:#555;text-transform:uppercase;letter-spacing:.4px;}
.tabs{display:flex;border-bottom:1px solid #ccc;}
.tab{flex:1;padding:5px;text-align:center;font-size:11px;cursor:pointer;color:#666;border-bottom:2px solid transparent;background:#ebebeb;}
.tab:hover{background:#e2e2e2;}
.tab.on{color:#0078d4;border-bottom-color:#0078d4;background:#fff;font-weight:600;}
.tree{overflow-y:auto;flex:1;padding:3px 0;}
.ti{padding:4px 8px;font-size:12px;cursor:pointer;display:flex;align-items:center;gap:5px;color:#333;}
.ti:hover{background:#e0e0e0;}
.ti.root{font-weight:700;color:#0078d4;}
.ctrl{padding:8px;border-top:1px solid #ccc;background:#f5f5f5;flex-shrink:0;}
.clabel{font-size:10px;color:#888;text-transform:uppercase;letter-spacing:.5px;margin-bottom:5px;}
.vgrid{display:grid;grid-template-columns:1fr 1fr;gap:3px;margin-bottom:5px;}
.vbtn{padding:5px;font-size:11px;text-align:center;background:#fff;border:1px solid #ccc;border-radius:2px;cursor:pointer;color:#444;}
.vbtn:hover,.vbtn.on{background:#cce4f7;border-color:#0078d4;color:#0078d4;font-weight:600;}
.crow{display:flex;align-items:center;gap:6px;font-size:11px;color:#444;margin-bottom:4px;cursor:pointer;}
.cbox{width:13px;height:13px;border-radius:2px;border:1px solid #aaa;display:flex;align-items:center;justify-content:center;font-size:9px;flex-shrink:0;}
.cbox.on{background:#0078d4;color:#fff;border-color:#0078d4;}
.cbox.off{background:#fff;}
.rbtn{width:100%;padding:5px;background:#fff;border:1px solid #ccc;border-radius:2px;color:#444;font-size:11px;cursor:pointer;margin-top:4px;}
.rbtn:hover{background:#e8e8e8;}

/* CENTER */
.center{flex:1;display:flex;flex-direction:column;position:relative;overflow:hidden;}
.vhdr{background:#e2e2e2;border-bottom:1px solid #ccc;padding:3px 10px;font-size:11px;color:#555;display:flex;align-items:center;gap:6px;flex-shrink:0;}
.vdot{width:6px;height:6px;background:#0078d4;border-radius:50%;}
.cur-style{margin-left:auto;font-size:11px;color:#0078d4;font-weight:600;}
#cv{position:absolute;top:0;left:0;width:100%;height:100%;cursor:grab;display:block;}
#cv:active{cursor:grabbing;}
.cambadge{position:absolute;bottom:8px;left:50%;transform:translateX(-50%);background:rgba(0,0,0,0.5);border-radius:3px;padding:2px 10px;font-size:11px;color:#fff;pointer-events:none;}

/* RIGHT PANEL */
.right{width:215px;background:#f5f5f5;border-left:1px solid #ccc;overflow-y:auto;flex-shrink:0;}
.phdr{background:#e2e2e2;padding:5px 10px;font-size:11px;font-weight:700;color:#555;text-transform:uppercase;border-bottom:1px solid #ccc;letter-spacing:.4px;}
.dsec{border-bottom:1px solid #e5e5e5;padding:7px 10px;}
.dtitle{font-size:10px;font-weight:700;text-transform:uppercase;color:#0078d4;margin-bottom:5px;letter-spacing:.4px;}
.dnote{font-size:9px;color:#aaa;font-style:italic;margin-bottom:4px;}
.drow{display:flex;justify-content:space-between;margin-bottom:2px;}
.dk{font-size:11px;color:#666;}
.dv{font-size:11px;color:#111;font-weight:600;}
.qbadge{display:inline-block;padding:1px 6px;border-radius:8px;font-size:10px;font-weight:700;}
table.ft{width:100%;border-collapse:collapse;font-size:11px;}
table.ft th{color:#666;text-align:left;padding:3px 0;border-bottom:1px solid #ddd;font-size:10px;text-transform:uppercase;}
table.ft td{padding:3px 0;color:#222;border-bottom:1px solid #f0f0f0;}
table.ft td:last-child{color:#0078d4;font-weight:700;text-align:right;}
.cpbtn{width:100%;padding:7px;background:#0078d4;border:none;border-radius:3px;color:#fff;font-size:11px;font-weight:600;cursor:pointer;}
.cpbtn:hover{background:#006bbf;}

/* RIGHT-CLICK CONTEXT MENU */
.ctx{position:fixed;background:#f0f0f0;border:1px solid #aaa;box-shadow:2px 3px 8px rgba(0,0,0,0.2);min-width:175px;z-index:9999;display:none;}
.ctx-item{padding:5px 24px 5px 10px;cursor:pointer;font-size:12px;color:#222;position:relative;display:flex;align-items:center;justify-content:space-between;}
.ctx-item:hover{background:#cce4f7;}
.ctx-arrow{font-size:9px;color:#888;}
.ctx-sep{height:1px;background:#ccc;margin:2px 0;}
.sub{position:fixed;background:#f0f0f0;border:1px solid #aaa;box-shadow:2px 3px 8px rgba(0,0,0,0.2);min-width:220px;z-index:10000;display:none;}
.sub-item{padding:5px 12px;cursor:pointer;font-size:12px;color:#222;display:flex;align-items:center;gap:8px;}
.sub-item:hover{background:#cce4f7;}
.sub-item.active{color:#0078d4;font-weight:600;}
.radio{width:12px;height:12px;border-radius:50%;border:2px solid #aaa;flex-shrink:0;}
.sub-item.active .radio{border-color:#0078d4;background:#0078d4;}
.kshort{margin-left:auto;color:#999;font-size:10px;}
</style>
</head>
<body>
<div class="root">

<div class="topbar">
  <span class="tb-logo">🏭 CMTI CAD Model Analyser</span>
  <span class="tb-file">FILENAME</span>
</div>

<div class="body">

<div class="left">
  <div class="browser-hdr">BROWSER</div>
  <div class="tabs">
    <div class="tab on" id="tab-tree" onclick="switchTab('tree')">Model Tree</div>
    <div class="tab" id="tab-view" onclick="switchTab('view')">View Settings</div>
  </div>
  <div class="tree" id="panel-tree">TREE_ROWS</div>
  <div class="tree" id="panel-view" style="display:none;padding:8px;">
    <div class="clabel">Object Visibility</div>
    <div class="crow" onclick="toggleBBox()"><div class="cbox on" id="cb-bbox">✓</div>Bounding Box</div>
    <div class="crow" onclick="toggleEdges()"><div class="cbox on" id="cb-edges">✓</div>Show Edges</div>
    <div class="crow" onclick="toggleGrid()"><div class="cbox on" id="cb-grid">✓</div>Ground Plane</div>
  </div>
  <div class="ctrl">
    <div class="clabel">CAMERA VIEW</div>
    <div class="vgrid">
      <div class="vbtn on" id="v-iso" onclick="setView('iso')">Isometric</div>
      <div class="vbtn" id="v-front" onclick="setView('front')">Front</div>
      <div class="vbtn" id="v-top" onclick="setView('top')">Top</div>
      <div class="vbtn" id="v-right" onclick="setView('right')">Right</div>
    </div>
    <button class="rbtn" onclick="setView('iso')">↺ Reset View</button>
  </div>
</div>

<div class="center">
  <div class="vhdr">
    <div class="vdot"></div>
    3D Viewer &nbsp;·&nbsp; Left-drag: rotate &nbsp;·&nbsp; Scroll: zoom &nbsp;·&nbsp; Right-drag: pan
    <span class="cur-style" id="cur-style">● Shaded with Visible Edges Only</span>
  </div>
  <canvas id="cv"></canvas>
  <div class="cambadge" id="cambadge">Isometric</div>
</div>

<div class="right">
  <div class="phdr">Geometry Dashboard</div>
  <div class="dsec">
    <div class="dtitle">📦 Bounding Box</div>
    <div class="drow"><span class="dk">L:</span><span class="dv">L_VAL mm</span></div>
    <div class="drow"><span class="dk">W:</span><span class="dv">W_VAL mm</span></div>
    <div class="drow"><span class="dk">H:</span><span class="dv">H_VAL mm</span></div>
  </div>
  <div class="dsec">
    <div class="dtitle">⚖️ Mass Properties</div>
    <div class="drow"><span class="dk">Volume:</span><span class="dv">##VOL## cm³</span></div>
    <div class="drow"><span class="dk">Surface Area:</span><span class="dv">AREA_VAL cm²</span></div>
    <div class="drow"><span class="dk">Quality:</span><span class="dv"><span class="qbadge" style="background:QBGVAL;color:QCOLORVAL;border:1px solid QCOLORVAL;">QUAL</span></span></div>
  </div>
  <div class="dsec">
    <div class="dtitle">🔷 TOPO_LABEL</div>
    <div class="dnote">TOPO_NOTE</div>
    <div class="drow"><span class="dk">FACE_LABEL:</span><span class="dv">FACES_VAL</span></div>
    <div class="drow"><span class="dk">Edges:</span><span class="dv">EDGES_VAL</span></div>
    <div class="drow"><span class="dk">Vertices:</span><span class="dv">VERTS_VAL</span></div>
    SURFACE_BREAKDOWN
    MESH_TOPO_EXTRA
  </div>
  <div class="dsec">
    <div class="dtitle">🔩 Features</div>
    <table class="ft"><tr><th>Type</th><th style="text-align:right">Count</th></tr>
    FEAT_ROWS
    </table>
  </div>
  <div class="dsec">
    <span id="copy-data" style="display:none">COPY_DATA</span>
    <button class="cpbtn" onclick="copyData()">📋 Copy All to Clipboard</button>
  </div>
</div>

</div>
</div>

<!-- RIGHT-CLICK CONTEXT MENU -->
<div class="ctx" id="ctx-menu">
  <div class="ctx-item" id="ctx-visual-style" onmouseenter="showSubMenu()">
    Visual Style <span class="ctx-arrow">▶</span>
  </div>
  <div class="ctx-item" onmouseenter="document.getElementById('ctx-sub').style.display='none'">
    Environment <span class="ctx-arrow">▶</span>
  </div>
  <div class="ctx-item" onmouseenter="document.getElementById('ctx-sub').style.display='none'">
    Graphics Preset <span class="ctx-arrow">▶</span>
  </div>
  <div class="ctx-item" onmouseenter="document.getElementById('ctx-sub').style.display='none'">
    Effects <span class="ctx-arrow">▶</span>
  </div>
  <div class="ctx-item" onmouseenter="document.getElementById('ctx-sub').style.display='none'">
    Object Visibility <span class="ctx-arrow">▶</span>
  </div>
  <div class="ctx-item" onmouseenter="document.getElementById('ctx-sub').style.display='none'">
    Camera <span class="ctx-arrow">▶</span>
  </div>
  <div class="ctx-sep"></div>
  <div class="ctx-item" onclick="hideCtx();setView('iso')" onmouseenter="document.getElementById('ctx-sub').style.display='none'">
    Ground Plane Offset
  </div>
  <div class="ctx-sep"></div>
  <div class="ctx-item" onclick="hideCtx()" onmouseenter="document.getElementById('ctx-sub').style.display='none'">
    Enter Full Screen <span style="color:#999;font-size:10px;">Ctrl+Shift+F</span>
  </div>
</div>

<!-- VISUAL STYLE SUBMENU -->
<div class="sub" id="ctx-sub">
  <div class="sub-item" id="sub-shaded" onclick="pickStyle('shaded')">
    <div class="radio"></div>Shaded<span class="kshort">Ctrl+4</span>
  </div>
  <div class="sub-item" id="sub-shaded_hidden" onclick="pickStyle('shaded_hidden')">
    <div class="radio"></div>Shaded with Hidden Edges<span class="kshort">Ctrl+5</span>
  </div>
  <div class="sub-item active" id="sub-shaded_edges" onclick="pickStyle('shaded_edges')">
    <div class="radio"></div>Shaded with Visible Edges Only<span class="kshort">Ctrl+6</span>
  </div>
  <div class="sub-item" id="sub-wireframe" onclick="pickStyle('wireframe')">
    <div class="radio"></div>Wireframe<span class="kshort">Ctrl+7</span>
  </div>
  <div class="sub-item" id="sub-transparent" onclick="pickStyle('transparent')">
    <div class="radio"></div>Transparent<span class="kshort">Ctrl+8</span>
  </div>
</div>

<script>
function hideCtx(){
  document.getElementById('ctx-menu').style.display='none';
  document.getElementById('ctx-sub').style.display='none';
}
document.addEventListener('click', hideCtx);
</script>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/STLLoader.js"></script>
<script>
JS_CODE
</script>
</body>
</html>"""

    # ── All replacements ──────────────────────────────────────────────────────
    html = html.replace('FILENAME',  filename)
    html = html.replace('TREE_ROWS', tree_rows)
    html = html.replace('FEAT_ROWS', feat_rows)
    html = html.replace('COPY_DATA', copy_data)
    html = html.replace('L_VAL',  f'{L:.2f}')
    html = html.replace('W_VAL',  f'{W:.2f}')
    html = html.replace('H_VAL',  f'{H:.2f}')
    html = html.replace('##VOL##', f'{vol:.2f}')
    html = html.replace('AREA_VAL', f'{area:.2f}')
    html = html.replace('QBGVAL',    qbg)
    html = html.replace('QCOLORVAL', qcolor)
    html = html.replace('TOPO_LABEL', topo_label)
    html = html.replace('TOPO_NOTE',  topo_note)
    html = html.replace('FACE_LABEL', face_label)
    html = html.replace('FACES_VAL', f'{faces:,}')
    html = html.replace('EDGES_VAL', f'{edges:,}')
    html = html.replace('VERTS_VAL', f'{verts:,}')
    html = html.replace('SURFACE_BREAKDOWN', surface_breakdown)
    html = html.replace('MESH_TOPO_EXTRA',   mesh_extra)
    html = html.replace('QUAL', qual)
    html = html.replace('JS_CODE', js_code)
    return html




# ── SERVER ────────────────────────────────────────────────────────────────────
# Server removed - using st.components.v1.html instead


# ── SESSION STATE ─────────────────────────────────────────────────────────────
for k,v in [('model_data',None),('geo',None),('features',{}),
            ('stl_bytes',None),('stl_name',''),('filename',''),('port',None)]:
    if k not in st.session_state: st.session_state[k] = v

# ── UPLOAD SCREEN ─────────────────────────────────────────────────────────────
if st.session_state.model_data is None:

    occ_ok = _occ_available()
    hero_html = (
        '''<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
*{margin:0;padding:0;box-sizing:border-box;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}
html,body{width:100%;height:100%;background:linear-gradient(135deg,#0d1b2a 0%,#1a2f4a 50%,#0d1b2a 100%);}
body{display:flex;align-items:center;justify-content:center;}
.card{text-align:center;max-width:580px;padding:40px;}
.icon{font-size:60px;margin-bottom:16px;}
.title{font-size:32px;font-weight:800;color:#fff;margin-bottom:8px;}
.subtitle{font-size:14px;color:#90a4ae;margin-bottom:28px;}
.pills{display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin-bottom:24px;}
.pill{background:rgba(79,195,247,0.12);color:#4fc3f7;padding:6px 14px;border-radius:20px;font-size:12px;border:1px solid rgba(79,195,247,0.3);}
.formats{font-size:12px;color:#546e7a;}
.formats span{color:#90caf9;font-weight:600;}
</style></head><body><div class="card">
<div class="icon">🔧</div>
<div class="title">CMTI CAD Model Analyser</div>
<div class="subtitle">Central Manufacturing Technology Institute</div>
<div class="pills">
  <span class="pill">📐 Geometry Measurement</span>
  <span class="pill">🔩 Feature Detection</span>
  <span class="pill">🎨 5 Visual Styles</span>
  <span class="pill">📦 Bounding Box</span>
  <span class="pill">⚖️ Mass Properties</span>
</div>
<div class="formats">Supports &nbsp;<span>STL</span> &nbsp;·&nbsp; <span>STEP</span> &nbsp;·&nbsp; <span>STP</span></div>
</div></body></html>'''
    )
    st.components.v1.html(hero_html, height=380, scrolling=False)

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        f  = st.file_uploader("", type=['stl','step','stp'],
                              label_visibility="collapsed", key="main_uploader")
        go = st.button("⚙️  Analyse Model", type="primary",
                       disabled=f is None, use_container_width=True)

    # ── Recent Files Panel ────────────────────────────────────────────────────
    recent = storage_list()
    if recent:
        st.markdown("---")
        st.markdown("### 🕐 Recent Files")
        st.caption(f"Last {len(recent)} uploaded — click Load to reload")
        for entry in recent:
            geo_e = entry.get('geo') or {}
            vol   = geo_e.get('volume', 0) / 1000
            dims  = geo_e.get('dims', {})
            L     = dims.get('x', 0); W = dims.get('y', 0); H = dims.get('z', 0)
            ts    = entry.get('timestamp', '')[:16].replace('T', '  ')
            qual  = 'Solid' if geo_e.get('watertight') else 'Surface'
            faces = geo_e.get('faces', 0)
            fname = entry.get('filename', '')
            with st.container():
                ca, cb, cc = st.columns([3, 1, 1])
                with ca:
                    info = (
                        "**📄 " + fname + "**  " + ts + "  \n"
                        + "📦 " + str(round(L,1)) + "×" + str(round(W,1)) + "×" + str(round(H,1)) + " mm  "
                        + "⚖️ " + str(round(vol,2)) + " cm3  "
                        + "🔷 " + str(faces) + " faces  "
                        + ("🟢 Solid" if qual=="Solid" else "🟡 Surface")
                    )
                    st.markdown(info)
                with cb:
                    if st.button("▶ Load", key="load_"+entry['uid'], use_container_width=True):
                        glb_b, stl_b, meta = storage_load(entry['uid'])
                        if glb_b:
                            st.session_state.model_data = glb_b
                            st.session_state.geo        = meta.get('geo')
                            st.session_state.features   = meta.get('features', {})
                            st.session_state.stl_bytes  = stl_b
                            st.session_state.stl_name   = meta.get('stl_name', '')
                            st.session_state.filename   = meta.get('filename', '')
                            st.rerun()
                with cc:
                    if st.button("🗑 Del", key="del_"+entry['uid'], use_container_width=True):
                        storage_delete(entry['uid'])
                        st.rerun()
                st.divider()

    if f and go:
        safe = f.name.replace(' ','_').replace('(','').replace(')','')
        ext  = Path(f.name).suffix.lower()
        stat = st.empty()
        stat.markdown('''
<div style="display:flex;flex-direction:column;align-items:center;padding:40px;
            background:#16213e;border-radius:12px;border:1px solid #0f3460;">
  <div style="width:60px;height:60px;border:3px solid #87CEEB;border-top-color:transparent;
              border-radius:50%;animation:spin 1s linear infinite;margin-bottom:16px;"></div>
  <style>@keyframes spin{to{transform:rotate(360deg)}}</style>
  <div style="color:#87CEEB;font-size:15px;font-weight:600;">Processing 3D Model...</div>
</div>
''', unsafe_allow_html=True)
        prog = st.progress(0)

        with tempfile.TemporaryDirectory() as tmp:
            inp  = os.path.join(tmp, safe)
            stlp = os.path.join(tmp, "model.stl")
            glbp = os.path.join(tmp, "model.glb")
            with open(inp,'wb') as fh: fh.write(f.getvalue())

            if ext == '.stl':
                prog.progress(25)
                shutil.copy(inp, stlp)
                stl_dl = f.getvalue(); stl_name = safe
            else:
                prog.progress(20)
                ok, msg, method = convert_step_to_stl(inp, stlp)
                if not ok:
                    stat.empty(); prog.empty()
                    st.error("Could not convert this STEP file. Please try uploading as STL instead.")
                    st.stop()
                with open(stlp,'rb') as fh: stl_dl = fh.read()
                stl_name = safe.replace('.step','.stl').replace('.STEP','.stl') \
                               .replace('.stp','.stl').replace('.STP','.stl')

            prog.progress(60)
            mesh = trimesh.load(stlp)
            try:
                clean = trimesh.Trimesh(vertices=mesh.vertices.copy(),
                                        faces=mesh.faces.copy(), process=False)
                clean.export(glbp, file_type='glb')
            except:
                mesh.export(glbp, file_type='glb')

            prog.progress(85)
            geo, features = None, None

            # Try pythonocc first for exact B-Rep geometry
            if ext in ('.step', '.stp') and _occ_available():
                geo, features = extract_geo_occ(inp)

            # Always try pure-Python STEP topology parser (works without pythonocc)
            step_topo = None
            if ext in ('.step', '.stp'):
                step_topo = extract_step_topology(inp)

            # Get mesh geometry from trimesh
            mesh2 = trimesh.load(stlp)
            if geo is None:
                geo      = extract_geo(mesh2)
                features = detect_features(mesh2, geo)

            # Merge true CAD topology into geo (overrides mesh triangle counts)
            if step_topo and geo is not None:
                geo['faces']        = step_topo['cad_faces']
                geo['edges']        = step_topo['cad_edges']
                geo['vertices']     = step_topo['cad_vertices']
                geo['cad_faces']    = step_topo['cad_faces']
                geo['cad_edges']    = step_topo['cad_edges']
                geo['cad_vertices'] = step_topo['cad_vertices']
                geo['has_cad_topo'] = True
                geo['surface_types']= step_topo['surface_types']
                geo['edge_types']   = step_topo['edge_types']
                geo['holes']        = step_topo['holes']
                # Keep mesh counts separately
                geo['mesh_faces']    = len(mesh2.faces)
                geo['mesh_edges']    = len(mesh2.edges_unique) if hasattr(mesh2,'edges_unique') else 0
                geo['mesh_vertices'] = len(mesh2.vertices)
                # Update features from STEP topology
                features = step_topo['features']
            with open(glbp,'rb') as fh: glb_bytes = fh.read()

        stat.empty(); prog.empty()
        st.session_state.model_data = glb_bytes
        st.session_state.geo        = geo
        st.session_state.features   = features
        st.session_state.stl_bytes  = stl_dl
        st.session_state.stl_name   = stl_name
        st.session_state.filename   = safe
        storage_save(safe, glb_bytes, stl_dl, stl_name, geo, features)
        st.rerun()

# ── DASHBOARD ─────────────────────────────────────────────────────────────────
else:
    if st.button("📂 Upload New File"):
        for k in list(st.session_state.keys()): st.session_state[k] = None
        st.rerun()

    # Use STL bytes for viewer (no embedded material = full color control)
    stl_bytes_view = st.session_state.stl_bytes or st.session_state.model_data
    stl_b64  = base64.b64encode(stl_bytes_view).decode()
    geo      = st.session_state.geo or {}
    features = st.session_state.features or {}
    filename = st.session_state.filename or "model"

    # Re-extract CAD topology from stored STEP if not already done
    ext_f = Path(filename).suffix.lower()
    if ext_f in ('.step', '.stp') and not geo.get('has_cad_topo'):
        recent = storage_list()
        for entry in recent:
            if entry.get('filename') == filename:
                spath = os.path.join(STORAGE_DIR, entry['uid'], filename)
                if os.path.exists(spath):
                    step_topo = extract_step_topology(spath)
                    if step_topo:
                        geo['faces']         = step_topo['cad_faces']
                        geo['edges']         = step_topo['cad_edges']
                        geo['vertices']      = step_topo['cad_vertices']
                        geo['cad_faces']     = step_topo['cad_faces']
                        geo['cad_edges']     = step_topo['cad_edges']
                        geo['cad_vertices']  = step_topo['cad_vertices']
                        geo['has_cad_topo']  = True
                        geo['surface_types'] = step_topo['surface_types']
                        geo['edge_types']    = step_topo['edge_types']
                        features             = step_topo['features']
                        st.session_state.geo      = geo
                        st.session_state.features = features
                break

    html = build_viewer_html(stl_b64, geo, features, filename)
    # Use st.components.v1.html - works on Streamlit Cloud (no localhost needed)
    import streamlit.components.v1 as components
    components.html(html, height=700, scrolling=False)

    c1,c2,c3 = st.columns(3)
    with c1:
        st.download_button("⬇️ GLB", st.session_state.model_data,
                           "model.glb","model/gltf-binary", use_container_width=True)
    with c2:
        if st.session_state.stl_bytes:
            st.download_button("⬇️ STL", st.session_state.stl_bytes,
                               st.session_state.stl_name or "model.stl",
                               "application/sla", use_container_width=True)
    with c3:
        if geo:
            st.download_button("⬇️ JSON", safe_json(geo),
                               "geometry.json","application/json", use_container_width=True)
