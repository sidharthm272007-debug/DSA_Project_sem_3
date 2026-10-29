# ResQGraph: disaster-response routing and simulation for Coimbatore

One model of the real Coimbatore road network, for people on the road and people watching a disaster unfold:

* **Best route now:** pick what has happened, get a recommended route, the alternatives it beat and why, the nearest hospital that is actually reachable, and the roads the city is leaning on. Add the closures you know about and they are taken into account.
* **Disaster lab:** combine any hazards, set the hour and the weather, and run a minute-by-minute simulation — damage, dispatch, re-routing, hospital outages, debris clearing — on the real road network.

Every recommendation carries its reasons (numbers computed from the model) and what it rests on (measured, approximate, assumed, missing).

## Run

You need Python 3.10 to 3.13 ([python.org](https://www.python.org/downloads/); on Windows tick "Add python.exe to PATH" during install). The app itself needs well under 1 GB of free memory.

**First, be in the right folder.** After unzipping you should see `app.py`, `requirements.txt` and `run_windows.bat` right there (type `dir` on Windows, `ls` elsewhere). If you only see a folder called `app`, go into it first (`cd app`).

**Windows:** double-click **`run_windows.bat`**. It installs what the app needs (first time only, a few minutes) and starts it; your browser opens by itself.

**Or by hand, any system** (open a terminal *in the folder that contains `app.py`*; folder names with spaces or brackets are fine):

```bash
python -m pip install -r requirements.txt        # first time only
python -m streamlit run app.py                   # not "python app.py"
```

On macOS / Linux, `bash run_mac_linux.sh` does the same. Then open the Local URL it prints (usually http://localhost:8501).

Common problems:

* *"Could not open requirements file" or "File does not exist: app.py"*: the terminal is in the wrong folder. Go to the folder that contains `app.py` and `requirements.txt` and run the commands again.
* *"Defaulting to user installation because normal site-packages is not writeable"*: normal, not an error; pip is installing for your user account.
* *"No module named yaml" / "No module named ..."*: the install step was skipped. Run the `pip install` line above in the folder that contains `requirements.txt`.
* *The screen goes blank, freezes, or the app crawls*: pick **Light** or **No maps** in the sidebar, or start with `run_windows_no_maps.bat`. If you see red error text instead of the page, copy its first line.
* *A warning about "run it with streamlit run"*: the app was started with plain `python app.py`. Use `python -m streamlit run app.py`.
* *`streamlit` is not recognised*: use `python -m streamlit run app.py` (or `py -m streamlit run app.py` on Windows).

For developers: `python -m pip install -r requirements-dev.txt`, then `python -m pytest` (automated tests, including whole-app runs on the real city) and `python -m scripts.verify` (end-to-end checks on the real data, ~20 s; see `VERIFICATION.txt`). The full suite passes on the oldest package versions in `requirements.txt` and on current ones.

## Pages

| Page | What it does |
|---|---|
| Best route now | Route advice for a chosen situation; "nearest working hospital" mode; your own local knowledge; the roads people depend on |
| Disaster lab | Combine earthquake, fire, gas leak, flood, bridge failure, storm, accident or a custom event; watch dispatch, re-routing, hospital outages and debris clearing minute by minute |
| Data & assumptions | What is measured, approximate, assumed or missing; every constant in `assumptions.yaml` |

## How the route recommendation is made

**Route advice** (`routing/`). The scenario's damage (closed and slowed roads, exposure to the hazard, hospitals out of service) is turned into a risk score for every stretch of road. Candidate routes come from Yen's algorithm (`routing/kshortest.py`) -- the K genuinely different shortest routes by pure time -- plus one further search biased hard toward safety, so the safest extreme of the trade-off is represented even if it needs a detour Yen's-by-time alone would not rank highly. That pool is narrowed to its Pareto-optimal front (`routing/pareto.py`): the routes where no other candidate is both faster and safer. Minimising a single blended number directly, without finding the front first, can skip a genuinely best route on a concave stretch of the trade-off curve -- a provable limitation of that approach, not a tuning problem. The recommendation is the point on the front minimising `time + v x risk-km`, where `v` is how many minutes you would add to avoid a kilometre of full-risk road (cautious / balanced / fastest). Reported times are always true times on the damaged network. Roads you know are blocked are removed; roads you know are unsafe get maximum risk. If your start is cut off, it finds the nearest junction with an open route within walking distance. It is an optimiser with an explanation, not a learned model: there is no user data to learn from and none is invented. Yen's algorithm runs on the full road network -- an earlier, geographically-restricted version of it was found (and proven, against an independent networkx cross-check) to occasionally return a wrong answer, so it was removed in favour of a slower but always-correct search; a typical trip resolves in well under a second, a long cross-city one can take several.

**Roads people depend on** (`routing/lifelines.py`). Each home's fastest route to its nearest open hospital is traced once (exact shortest-path forest); a road's load is the people whose route uses it. Compared with a normal day this shows lifelines, surging detours and cut roads.

## Known limits (also shown in the app)

* **Fire stations:** only 2 are mapped in OpenStreetMap for this area, so fire-response times and unserved fire calls in the Disaster lab run worse than the real city until at least 8 are mapped. Add stations to `data/coimbatore/facilities_extra.csv`, run `python -m scripts.build_derived`, and they switch on.
* **Roads** are OpenStreetMap roads only: service, campus and informal roads are not loaded and railways act as barriers.
* **Population** is a road-density proxy scaled to the city estimate, not a census grid. **Ambulance bases** are assumed. **Hazard parameters** are heuristic, not calibrated to a specific event.
* **Congestion feedback** is not modelled: route times do not include the queues that many people re-routing at once would cause.
* **Resources:** the app holds the road network in memory, about 0.25 GB after opening and peaking around 0.35 GB during a simulation run, so keep 0.5 GB free for comfort.
* **Maps and slow or blank screens.** The maps are drawn by your browser's graphics (WebGL). On a laptop with weak or driver-troubled graphics, heavy 3D scenes can freeze the browser or blank the screen. So the default is **Light** (flat 2D, fewer shapes, no labels). The sidebar's **Map detail** switch offers **Rich** (3D, for a good graphics card) and **No maps** (text and tables only; every result is still there). `run_windows_no_maps.bat` starts with maps off. Measured with graphics drawn in software (a laptop with no usable GPU): panning Rich runs about 3 frames per second, Light about 7, No maps is unaffected. Also: use current Chrome or Edge with hardware acceleration on, update the graphics driver, close other tabs.
* **Maps** load street tiles from CARTO's public server, so the browser needs internet access for the basemap; without it the analysis layers still draw on a plain dark background.

## Layout

```
app.py                      entry point (checks the install, then starts navigation)
run_windows.bat / run_windows_no_maps.bat / run_mac_linux.sh   one-step launchers
requirements.txt / requirements-dev.txt
ui/                         pages, components, scenario presets
viz/                        map layers, design tokens (theme.py), animation
routing/                    risk model, route recommender, lifelines (see table below)
sim/                        city model, hazards, discrete-event engine, repair planning, reports (see table below)
store/                      road graph (CSR) and spatial/KD-tree index
data/coimbatore/            OpenStreetMap-derived data and facility lists
scripts/                    build_derived (data pipeline), verify (end-to-end checks)
tests/                      unit tests against networkx / brute force, and whole-app tests
assumptions.yaml            every constant, in one place
```

### Finding a specific algorithm or behaviour

Every distinct piece of logic -- one algorithm, one simulated action -- has its own file, so a question like *"how does
dispatch decide which ambulance to send?"* has one file to open, not a search across a monolith:

| Ask about... | Look in... |
|---|---|
| Yen's K-shortest-paths algorithm | `routing/kshortest.py` |
| Pareto frontier / multi-objective trade-off | `routing/pareto.py` |
| How a route is scored, recommended end to end | `routing/recommend.py` |
| Per-road risk score from the active hazards | `routing/risk.py` |
| "Roads people depend on" / lifelines | `routing/lifelines.py` |
| Compressed Sparse Row (CSR) graph structure | `store/csr_graph.py` |
| KD-tree nearest-road snapping | `store/spatial_index.py` |
| The top-level per-minute simulation loop | `sim/engine.py` (`Simulation.run`) |
| Hazard -> road/network state update, metrics recording | `sim/engine_state.py` |
| Incident (911-call) generation, expiry | `sim/engine_incidents.py` |
| Unit dispatch / nearest-idle-unit assignment | `sim/engine_dispatch.py` |
| Per-minute movement, live re-routing, arrivals | `sim/engine_movement.py` |
| Repair-crew job assignment | `sim/engine_repair.py` (ranking itself: `sim/repair.py`, Kruskal/Union-Find) |
| A specific hazard type (earthquake, fire, flood, ...) | `sim/hazards/<name>.py` -- one file per hazard, shared contract in `sim/hazards/base.py` |
| Dijkstra-based routing (point-to-point, reachability, nearest-of-many) | `sim/router.py` |
| Plain-language findings text, summary stats | `sim/report.py`, `sim/metrics.py` |
| City data loading (graph, facilities, zones, water) | `sim/city.py` |

Each of those files opens with a short docstring describing what it owns and, for the `sim/engine_*.py` files, what
question a reader ("if a professor asks...") it is meant to answer.
