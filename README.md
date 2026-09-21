# ResQGraph: emergency access and infrastructure planning for Coimbatore

One model of the real Coimbatore road network, two kinds of user:

* **Someone on the road in an emergency**, or who knows which roads work: pick what has happened, get a recommended route, the alternatives it beat and why, the nearest hospital that is actually reachable, and the roads the city is leaning on. Add the closures you know about and they are taken into account.
* **Someone who plans or funds infrastructure**: where the next hospital, fire station, ambulance post, road or bridge would help most; a verdict on any idea you put on the map; a back-test of whether flyovers and hospitals that already exist are worth it; a downloadable planning brief.

Every recommendation carries its reasons (numbers computed from the model) and what it rests on (measured, approximate, assumed, missing).

## Run

You need Python 3.10 to 3.13 ([python.org](https://www.python.org/downloads/); on Windows tick "Add python.exe to PATH" during install). The app itself needs about 1 GB of free memory.

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

For developers: `python -m pip install -r requirements-dev.txt`, then `python -m pytest` (automated tests, including whole-app runs on the real city) and `python -m scripts.verify` (end-to-end checks on the real data, ~2 min; see `VERIFICATION.txt`). The full suite passes on the oldest package versions in `requirements.txt` and on current ones.

## Pages

| Page | For | What it does |
|---|---|---|
| Best route now | anyone | Route advice for a chosen situation; "nearest working hospital" mode; your own local knowledge; the roads people depend on |
| Disaster lab | responders, planners | Combine earthquake, fire, gas leak, flood, bridge failure, storm, accident or a custom event; watch dispatch, re-routing, hospital outages and debris clearing minute by minute |
| Where to build | planners | Ranked sites for a hospital / fire station / ambulance post; ranked road and bridge ideas; planning brief |
| Test an idea | planners | Put your own facility or road on the map and get a verdict, who it helps in which situation, and a nearby better spot if one exists |
| Is it useful? | planners | Remove an existing flyover, bridge or hospital and re-score everything, including the Avinashi Road elevated corridor |
| Data & assumptions | everyone | What is measured, approximate, assumed or missing; every constant in `assumptions.yaml` |

## How the recommendations are made

**Route advice** (`routing/`). The scenario's damage (closed and slowed roads, exposure to the hazard, hospitals out of service) is turned into a risk score for every stretch of road. Several genuinely different routes are found by shortest-path search on `time x (1 + penalty x risk)` for a range of penalties; reported times are always true times on the damaged network. The recommendation minimises `time + v x risk-km`, where `v` is how many minutes you would add to avoid a kilometre of full-risk road (cautious / balanced / fastest). Roads you know are blocked are removed; roads you know are unsafe get maximum risk. If your start is cut off, it finds the nearest junction with an open route within walking distance. It is an optimiser with an explanation, not a learned model: there is no user data to learn from and none is invented.

**Roads people depend on** (`routing/lifelines.py`). Each home's fastest route to its nearest open hospital is traced once (exact shortest-path forest); a road's load is the people whose route uses it. Compared with a normal day this shows lifelines, surging detours and cut roads.

**Planning advice** (`advisor/`). Every idea is judged against the same portfolio of scenarios: a normal day plus the worst moments of earthquakes, fires, floods, bridge failures and a storm, weighted by priorities you set. New facilities and new roads use exact incremental shortest-path updates (checked against rebuilding the whole graph); existing infrastructure is judged by removing it. Verdicts come from thresholds in `assumptions.yaml`.

## Known limits (also shown in the app)

* **Fire stations:** only 2 are mapped in OpenStreetMap for this area, so fire-service numbers are left out of headline advisor results until at least 8 are mapped. Add stations to `data/coimbatore/facilities_extra.csv`, run `python -m scripts.build_derived`, and they switch on.
* **Costs** are placeholders except the press-reported ₹177 crore/km flyover figure. Benefits do not depend on cost; value-for-money numbers do.
* **Roads** are OpenStreetMap roads only: service, campus and informal roads are not loaded and railways act as barriers, so a suggested new road may just patch missing data. Check on the ground.
* **Population** is a road-density proxy scaled to the city estimate, not a census grid. **Ambulance bases** are assumed. **Hazard parameters** are heuristic, not calibrated to a specific event.
* **Congestion feedback** is not modelled: route times do not include the queues that many people re-routing at once would cause.
* **Resources:** the app holds the road network in memory. Measured: about 0.3 GB after opening, about 0.5 GB once the planning pages have built their scenario set, and a peak of about 0.6 GB during the heaviest search, so keep 1 GB free. The first visit to a planning page waits for a one-off ~15 s set-up behind a message. Each user gets their own priority settings; the heavy data is shared.
* **Maps and slow or blank screens.** The maps are drawn by your browser's graphics (WebGL). On a laptop with weak or driver-troubled graphics, heavy 3D scenes can freeze the browser or blank the screen. So the default is **Light** (flat 2D, fewer shapes, no labels). The sidebar's **Map detail** switch offers **Rich** (3D, for a good graphics card) and **No maps** (text and tables only; every result is still there). `run_windows_no_maps.bat` starts with maps off. Measured with graphics drawn in software (a laptop with no usable GPU): panning Rich runs about 3 frames per second, Light about 7, No maps is unaffected. Also: use current Chrome or Edge with hardware acceleration on, update the graphics driver, close other tabs.
* **Maps** load street tiles from CARTO's public server, so the browser needs internet access for the basemap; without it the analysis layers still draw on a plain dark background.
* Counts of "people inside the standard" cross a time threshold, so they can swing in a single scenario; the minutes-saved figure is steadier.

## Layout

```
app.py                      entry point (checks the install, then starts navigation)
run_windows.bat / run_windows_no_maps.bat / run_mac_linux.sh   one-step launchers
requirements.txt / requirements-dev.txt
ui/                         pages, components, scenario presets
viz/                        map layers and animation
routing/                    risk model, route recommender, lifelines
advisor/                    portfolio, evaluator, siting, road ideas, verdicts, planning brief
sim/                        city model, hazards, discrete-event engine, repair planning, reports
store/  traffic/            road graph (CSR) and junction geometry
data/coimbatore/            OpenStreetMap-derived data and facility lists
scripts/                    build_derived (data pipeline), verify (end-to-end checks)
tests/                      unit tests against networkx / brute force, and whole-app tests
assumptions.yaml            every constant, in one place
sample_planning_brief.md    the brief generated by the app on the real city
```
