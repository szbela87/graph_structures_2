import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import random
import math
from utils import generate_full_graph

# Wide layout
st.set_page_config(layout="wide")
#st.title("Skálafüggetlen Gráfok Vizualizációja")

#---- Define graph generators ----

def generate_barabasi_albert(n, m, seed=None):
    if seed is not None:
        return nx.barabasi_albert_graph(n=n, m=m, seed=seed)
    return nx.barabasi_albert_graph(n=n, m=m)

def generate_hierarchical(r, h):
    return nx.balanced_tree(r=r, h=h)

def generate_klemm_eguilluz(n, m):
    G = nx.complete_graph(m)
    active = list(range(m))
    for i in range(m, n):
        G.add_node(i)
        for j in active:
            G.add_edge(i, j)
        active.append(i)
        degrees = {node: G.degree(node) for node in active}
        weights = [1/degrees[node] for node in active]
        to_remove = random.choices(active, weights=weights, k=1)[0]
        active.remove(to_remove)
    return G

def generate_duplication_divergence(n, p, seed=None):
    if seed is not None:
        random.seed(seed)
    G = nx.complete_graph(2)
    for new_node in range(2, n):
        G.add_node(new_node)
        original = random.choice(list(G.nodes())[:-1])
        for neighbor in list(G.neighbors(original)):
            if random.random() < p:
                G.add_edge(new_node, neighbor)
        if G.degree(new_node) == 0:
            G.add_edge(new_node, original)
    return G

def generate_nonlinear_pa(n, m, alpha, seed=None):
    if seed is not None:
        random.seed(seed)
    G = nx.complete_graph(m+1)
    for i in range(m+1, n):
        G.add_node(i)
        degrees = {node: G.degree(node) for node in G.nodes()}
        weights = [degrees[node]**alpha for node in G.nodes()]
        total = sum(weights)
        probs = [w/total for w in weights]
        targets = set()
        while len(targets) < m:
            selected = random.choices(list(G.nodes()), weights=probs, k=1)[0]
            targets.add(selected)
        for t in targets:
            G.add_edge(i, t)
    return G

def generate_aging(n, m, decay, seed=None):
    if seed is not None:
        random.seed(seed)
    G = nx.complete_graph(m)
    age = {node: 0 for node in G.nodes()}
    for t in range(m, n):
        G.add_node(t)
        age[t] = 0
        attractiveness = {node: G.degree(node) * math.exp(-decay * age[node]) for node in G.nodes()}
        total = sum(attractiveness.values())
        probs = [attr / total for attr in attractiveness.values()]
        targets = set()
        nodes = list(attractiveness.keys())
        while len(targets) < m:
            choice = random.choices(nodes, weights=probs, k=1)[0]
            targets.add(choice)
        for target in targets:
            G.add_edge(t, target)
        for node in age:
            age[node] += 1
    return G


def generate_apollonian_by_nodes(n):
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2])
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    faces = [(0, 1, 2)]
    next_node = 3
    # Addig iterálunk, amíg el nem érjük az n csúcsot
    while G.number_of_nodes() < n and faces:
        u, v, w = faces.pop(0)
        G.add_node(next_node)
        G.add_edges_from([(next_node, u), (next_node, v), (next_node, w)])
        faces.extend([(u, v, next_node), (v, w, next_node), (w, u, next_node)])
        next_node += 1
    return G

def generate_sbm(sizes, p_in, p_out, seed=None):
    # sizes pl. [50, 50] (két közösség)
    n_blocks = len(sizes)
    p_matrix = [[p_in if i==j else p_out for j in range(n_blocks)] for i in range(n_blocks)]
    if seed is not None:
        return nx.stochastic_block_model(sizes, p_matrix, seed=seed)
    return nx.stochastic_block_model(sizes, p_matrix)

def generate_geometric(n, radius, seed=None):
    if seed is not None:
        return nx.random_geometric_graph(n, radius, seed=seed)
    return nx.random_geometric_graph(n, radius)

def generate_holme_kim(n, m, p, seed=None):
    if seed is not None:
        return nx.powerlaw_cluster_graph(n, m, p, seed=seed)
    return nx.powerlaw_cluster_graph(n, m, p)

def generate_watts_strogatz(n, k, p, seed=None):
    if seed is not None:
        return nx.watts_strogatz_graph(n, k, p, seed=seed)
    return nx.watts_strogatz_graph(n, k, p)

def generate_erdos_renyi(n, p, seed=None):
    if seed is not None:
        return nx.erdos_renyi_graph(n, p, seed=seed)
    return nx.erdos_renyi_graph(n, p)





# Detailed descriptions for each model
details = {
    "Barabási–Albert modell": (
        "**Mechanizmus:** Új csomópontok preferenciálisan csatlakoznak magas fokszámú csúcsokhoz (rich-get-richer).\n"
        "**Fokszám-eloszlás:** Hatványfüggvényes (P(k) ~ k^{-3}).\n"
        "**Tulajdonságok:** Kibővült hub-ok, kis átlagos úthossz, alacsony klaszterezettség."
    ),
    "Hierarchikus hálózat": (
        "**Mechanizmus:** Kiegyensúlyozott r-ágazású, h-szintű fa.\n"
        "**Szerkezet:** Moduláris, több szinten szervezett.\n"
        "**Tulajdonságok:** Magas helyi klaszterezettség, skálafüggetlen modulhierarchia."
    ),
    "Klemm–Eguíluz modell": (
        "**Mechanizmus:** Új csúcsok csak aktív csúcsokhoz kapcsolódnak; minden lépésben egy aktív inaktívvá válik.\n"
        "**Tulajdonságok:** Nagy klaszterezettség, kis világ tulajdonságok, skálafüggetlen eloszlás."
    ),
    "Duplication–Divergence modell": (
        "**Mechanizmus:** Csomópont duplikáció, majd az új csak egy részét örökli az eredeti éleinek.\n"
        "**Tulajdonságok:** Hosszú farok jellegű eloszlás, biológiai hálózatokra jellemző tulajdonságok változtathatók p-vel."
    ),
    "Nonlineáris PA modell": (
        "**Mechanizmus:** Preferencia ∝ k^α.\n"
        "**Hatás:** α<1 gyengébb hub-kiépülés; α>1 szuperhub-ok.\n"
        "**Eloszlás:** Rugalmas hosszú farok."
    ),
    "Aging modell": (
        "**Mechanizmus:** Vonzóerő ∝ degree·exp(-decay·age).\n"
        "**Tulajdonságok:** Időben korlátozott hub-növekedés, kiegyensúlyozottabb eloszlás, megtartja a hatványfarokot."
    ),
    "Apollóniai hálózat": (
        "**Mechanizmus:** Háromszögek iteratív felosztása új csúcsok beszúrásával.\n"
        "**Tulajdonságok:** Determinisztikus, nagyon magas klaszterezettség, fraktál jelleg, skálafüggetlen fokszám."
    ),
    "Erdős–Rényi modell": (
        "**Mechanizmus:** Minden él egy adott p valószínűséggel jön létre véletlenszerűen.\n"
        "**Fokszám-eloszlás:** Binomiális (nagy n-re közel Poisson).\n"
        "**Tulajdonságok:** Nincs hub, nincsenek közösségek, kontrollmodell."
    ),
    "Watts–Strogatz modell": (
        "**Mechanizmus:** Egy körgráf éleit bizonyos valószínűséggel áthuzalozzuk véletlenszerűen.\n"
        "**Tulajdonságok:** Kis világ tulajdonság (kis átl. úthossz), magas klaszterezettség.\n"
        "**Fokszám-eloszlás:** Nem skálafüggetlen."
    ),
    "Holme–Kim modell": (
        "**Mechanizmus:** Preferenciális kapcsolódás háromszögeléssel kombinálva (háromszögelés p valószínűséggel).\n"
        "**Tulajdonságok:** Skálafüggetlen, de jelentősen magasabb klaszterezettség, mint a sima Barabási–Albert modellnél."
    ),
    "Geometriai véletlen gráf": (
        "**Mechanizmus:** Véletlenszerűen elhelyezett csúcsok, él akkor van két csúcs között, ha a távolságuk kisebb, mint egy r küszöb.\n"
        "**Tulajdonságok:** Térbeli szerkezet, klasztereződés, modellez pl. szenzorhálózatokat."
    ),
    "Stochastic Block Model": (
        "**Mechanizmus:** Előre megadott blokkok (közösségek), a blokkokon belül és kívül más-más kapcsolódási valószínűség.\n"
        "**Tulajdonságok:** Erős közösségi szerkezet, nincs hub, nem skálafüggetlen, jól szemlélteti a modularitást."
    ),
}


# Sidebar controls
# Külön widget az SBM sizes paraméterhez
def sbm_sizes_widget():
    size1 = st.sidebar.slider("Blokk 1 mérete", 10, 250, 50)
    size2 = st.sidebar.slider("Blokk 2 mérete", 10, 250, 50)
    return [size1, size2]

models = {
    "Barabási–Albert modell": {
        "func": generate_barabasi_albert,
        "params": {
            "n": (st.sidebar.slider, {"label": "Csúcsok száma (n)", "min_value": 10, "max_value": 2000, "value": 100}),
            "m": (st.sidebar.slider, {"label": "Élek száma csúcsanként (m)", "min_value": 1, "max_value": 20, "value": 5}),
            "seed": (st.sidebar.number_input, {"label": "Random seed", "min_value": 0, "max_value": 1000, "value": 42})
        }
    },
    "Hierarchikus hálózat": {
        "func": generate_hierarchical,
        "params": {
            "r": (st.sidebar.slider, {"label": "Elágazási fok (r)", "min_value": 2, "max_value": 5, "value": 2}),
            "h": (st.sidebar.slider, {"label": "Szintek száma (h)", "min_value": 1, "max_value": 5, "value": 3})
        }
    },
    "Klemm–Eguíluz modell": {
        "func": generate_klemm_eguilluz,
        "params": {
            "n": (st.sidebar.slider, {"label": "Csúcsok száma (n)", "min_value": 10, "max_value": 2000, "value": 100}),
            "m": (st.sidebar.slider, {"label": "Kezdő csúcsok száma (m)", "min_value": 1, "max_value": 20, "value": 5})
        }
    },
    "Duplication–Divergence modell": {
        "func": generate_duplication_divergence,
        "params": {
            "n": (st.sidebar.slider, {"label": "Csúcsok száma (n)", "min_value": 10, "max_value": 2000, "value": 100}),
            "p": (st.sidebar.slider, {"label": "Megőrzési valószínűség (p)", "min_value": 0.0, "max_value": 1.0, "value": 0.5, "step": 0.05}),
            "seed": (st.sidebar.number_input, {"label": "Random seed", "min_value": 0, "max_value": 1000, "value": 42})
        }
    },
    "Nonlineáris PA modell": {
        "func": generate_nonlinear_pa,
        "params": {
            "n": (st.sidebar.slider, {"label": "Csúcsok száma (n)", "min_value": 10, "max_value": 2000, "value": 100}),
            "m": (st.sidebar.slider, {"label": "Élek száma csúcsanként (m)", "min_value": 1, "max_value": 20, "value": 3}),
            "alpha": (st.sidebar.slider, {"label": "Alpha (α)", "min_value": 0.0, "max_value": 3.0, "value": 1.5, "step": 0.1}),
            "seed": (st.sidebar.number_input, {"label": "Random seed", "min_value": 0, "max_value": 1000, "value": 42})
        }
    },
    "Aging modell": {
        "func": generate_aging,
        "params": {
            "n": (st.sidebar.slider, {"label": "Csúcsok száma (n)", "min_value": 10, "max_value": 2000, "value": 100}),
            "m": (st.sidebar.slider, {"label": "Élek száma csúcsanként (m)", "min_value": 1, "max_value": 20, "value": 3}),
            "decay": (st.sidebar.slider, {"label": "Decay faktor", "min_value": 0.0, "max_value": 1.0, "value": 0.01, "step": 0.01}),
            "seed": (st.sidebar.number_input, {"label": "Random seed", "min_value": 0, "max_value": 1000, "value": 42})
        }
    },
    "Apollóniai hálózat": {
        "func": generate_apollonian_by_nodes,
        "params": {
            "n": (st.sidebar.slider, {"label": "Csúcsok száma (n)", "min_value": 3, "max_value": 2000, "value": 10})
        }
    },
    "Erdős–Rényi modell": {
        "func": generate_erdos_renyi,
        "params": {
            "n": (st.sidebar.slider, {"label": "Csúcsok száma (n)", "min_value": 10, "max_value": 2000, "value": 100}),
            "p": (st.sidebar.slider, {"label": "Él valószínűség (p)", "min_value": 0.0, "max_value": 1.0, "value": 0.05, "step": 0.01}),
            "seed": (st.sidebar.number_input, {"label": "Random seed", "min_value": 0, "max_value": 1000, "value": 42})
        }
    },
    "Watts–Strogatz modell": {
        "func": generate_watts_strogatz,
        "params": {
            "n": (st.sidebar.slider, {"label": "Csúcsok száma (n)", "min_value": 10, "max_value": 2000, "value": 100}),
            "k": (st.sidebar.slider, {"label": "Szomszédok száma (k)", "min_value": 2, "max_value": 20, "value": 4, "step": 2}),
            "p": (st.sidebar.slider, {"label": "Áthuzalozás valószínűség (p)", "min_value": 0.0, "max_value": 1.0, "value": 0.1, "step": 0.01}),
            "seed": (st.sidebar.number_input, {"label": "Random seed", "min_value": 0, "max_value": 1000, "value": 42})
        }
    },
    "Holme–Kim modell": {
        "func": generate_holme_kim,
        "params": {
            "n": (st.sidebar.slider, {"label": "Csúcsok száma (n)", "min_value": 10, "max_value": 2000, "value": 100}),
            "m": (st.sidebar.slider, {"label": "Élek száma csúcsanként (m)", "min_value": 1, "max_value": 20, "value": 4}),
            "p": (st.sidebar.slider, {"label": "Háromszögelés valószínűség (p)", "min_value": 0.0, "max_value": 1.0, "value": 0.3, "step": 0.01}),
            "seed": (st.sidebar.number_input, {"label": "Random seed", "min_value": 0, "max_value": 1000, "value": 42})
        }
    },
    "Geometriai véletlen gráf": {
        "func": generate_geometric,
        "params": {
            "n": (st.sidebar.slider, {"label": "Csúcsok száma (n)", "min_value": 10, "max_value": 2000, "value": 100}),
            "radius": (st.sidebar.slider, {"label": "Kapcsolódási távolság (radius)", "min_value": 0.01, "max_value": 0.5, "value": 0.15, "step": 0.01}),
            "seed": (st.sidebar.number_input, {"label": "Random seed", "min_value": 0, "max_value": 1000, "value": 42})
        }
    },
    "Stochastic Block Model": {
        "func": generate_sbm,
        "params": {
            "sizes": (lambda **kwargs: sbm_sizes_widget(), {}),
            "p_in": (st.sidebar.slider, {"label": "Csoporton belüli él valószínűség (p_in)", "min_value": 0.0, "max_value": 1.0, "value": 0.1, "step": 0.01}),
            "p_out": (st.sidebar.slider, {"label": "Csoportok közötti él valószínűség (p_out)", "min_value": 0.0, "max_value": 1.0, "value": 0.01, "step": 0.01}),
            "seed": (st.sidebar.number_input, {"label": "Random seed", "min_value": 0, "max_value": 1000, "value": 42})
        }
    },
}

# Model selection
choice = st.sidebar.selectbox("Válaszd ki a modellt:", list(models.keys()))
config = models[choice]

# Collect parameters
kwargs = {}
for param, (widget, opts) in config["params"].items():
    kwargs[param] = widget(**opts)

# Create a unique key for the current graph configuration
graph_config_key = f"{choice}_{str(kwargs)}"

# Initialize session state for graph and position if not exists or if config changed
if 'graph_config_key' not in st.session_state or st.session_state.graph_config_key != graph_config_key:
    st.session_state.graph_config_key = graph_config_key
    st.session_state.G = config["func"](**kwargs)
    st.session_state.pos = nx.spring_layout(st.session_state.G)

# Use graph and position from session state
G = st.session_state.G
pos = st.session_state.pos

# Two-column layout
col1, col2 = st.columns([3,2])
with col1:
    #st.subheader("Grafikus megjelenítés")
    fig, ax = plt.subplots(figsize=(6,6))
    nx.draw(G, pos, ax=ax, node_size=50)
    # (Cím a jobb oldali oszlopban jelenik meg)
    st.pyplot(fig)
with col2:
    #st.subheader("Modell részletei")
    # Display the model name as a header
    st.markdown(f"### {choice}")
    # Display detailed description
    st.markdown(details[choice])

    # Generate Computational Graph Section
    st.markdown("---")

    # Get CNAME from function name
    func_name = config["func"].__name__
    if func_name.startswith("generate_"):
        CNAME = func_name[9:]  # Remove "generate_" prefix
    else:
        CNAME = func_name

    # Get SEED from kwargs (default to 42 if not present)
    SEED = kwargs.get('seed', 42)

    # Create form with expander
    with st.expander("Számítási gráf generálása", expanded=False):
        with st.form("comp_graph_form"):
            st.write("Számítási gráf paramétereinek beállítása:")

            # Display SEED (from model)
            st.write(f"**SEED:** {SEED} (modell konfigurációból)")

            # Display CNAME (automatically generated)
            st.write(f"**CNAME:** {CNAME} (modell névből)")

            # INPUT_NUM slider
            INPUT_NUM = st.slider("Bemeneti neuronok száma (INPUT_NUM)",
                                  min_value=10, max_value=2000, value=85)

            # ACT_TYPE slider
            ACT_TYPE = st.slider("Aktivációs típus (ACT_TYPE)", min_value=0, max_value=10, value=9)

            # P_IN slider
            P_IN = st.slider("Bemenet-Rezervoár kapcsolódási valószínűség (P_IN)",
                             min_value=0.0, max_value=1.0, value=0.2, step=0.01)

            # P_OUT slider
            P_OUT = st.slider("Rezervoár-Kimenet kapcsolódási valószínűség (P_OUT)",
                              min_value=0.0, max_value=1.0, value=0.2, step=0.01)

            # FF checkbox
            FF = st.checkbox("Előrecsatolt (FF)", value=True)

            # RES_DIRECTED checkbox
            directed = st.checkbox("Irányított", value=True)
            RES_DIRECTED = "directed" if directed else "undirected"

            # DRAW checkbox
            DRAW = st.checkbox("Gráf vizualizáció generálása (DRAW)", value=False)

            # Submit button
            submitted = st.form_submit_button("Gráf generálása")

            if submitted:
                # Validate parameters
                if RES_DIRECTED == "undirected" and FF == True:
                    st.error("Hiba: FF nem lehet igaz, ha RES_DIRECTED 'undirected'")
                else:
                    try:
                        # Call generate_full_graph function
                        file_paths = generate_full_graph(
                            G=G,
                            INPUT_NUM=INPUT_NUM,
                            RES_DIRECTED=RES_DIRECTED,
                            FF=FF,
                            P_IN=P_IN,
                            P_OUT=P_OUT,
                            CNAME=CNAME,
                            ACT_TYPE=ACT_TYPE,
                            SEED=SEED,
                            pos=pos,
                            DRAW=DRAW
                        )

                        # Display success message and file paths
                        st.success("A számítási gráf sikeresen legenerálva!")

                        # Display trainable parameters
                        trainable_params = file_paths.get('trainable_parameters')
                        if trainable_params is not None:
                            st.write(f"**Tanítható paraméterek:** {trainable_params}")

                        st.write("Generált fájlok:")
                        for file_type, file_path in file_paths.items():
                            if file_type != 'trainable_parameters':  # Skip trainable_parameters in file list
                                st.write(f"- **{file_type}:** `{file_path}`")

                    except Exception as e:
                        st.error(f"Hiba a gráf generálása során: {str(e)}")
