# 🎬 Demo video script — *Sensorless Traffic* (target ≤ 5:00)

**Format:** screen recording of the live app + voice-over. English.
**Team:** Imad, Raúl & Nouh — split the narration (one section each) so it reads
as a team video; use "we / our team" throughout.
**Tip:** rehearse once; the timings below leave ~15 s of slack. Speak at a calm
pace — it is better to land at 4:50 than to rush.

---

### 0:00 – 0:35 · Hook, team & the city problem
> "Picture a city traffic department deciding where to add a bus lane or calm a
> street. To do it well, they need to know how busy each street is. The catch?
> A city only *knows* the traffic on streets where it installed a sensor — and in
> Valencia that's barely **2.4 %** of them. The other **97.6 % are blind spots**,
> and a nearby town like Paiporta has **no sensors at all**."
>
> "We're **Imad, Raúl and Nouh**, and our app **Sensorless Traffic** turns that
> tiny measured fraction into a congestion map for *every* street — even in towns
> that have never had a single sensor."

🖥️ *Open the app on the Congestion Map tab, the headline metrics visible. Let the
map finish rendering before you start talking over it.*

---

### 0:35 – 1:25 · Tab 1 — the blind-spot problem, made visible
> "This is Valencia's entire road network — about 54,000 street segments. By
> default we show our **model's congestion estimate for the whole city**: green
> is free-flowing, red is arterial-level demand."

🖥️ *Hover a couple of streets to show the tooltip (name + level). Note the
on-map legend, bottom-right.*

> "Now watch what the city *actually* measures."

🖥️ *Switch the layer to "Sensor coverage only". The map goes mostly grey.*

> "Everything grey is a blind spot — only these few coloured streets have a
> sensor. Our goal is to fill in all the grey, reliably."

🖥️ *Switch back to "Model estimate".*

---

### 1:25 – 2:35 · Tab 4 — methodology & how well it works
> "How? A full data-science pipeline — the CRISP-DM process from the course. We
> pull the road network from **OpenStreetMap** and hourly counts from
> **Valencia's loop sensors**, then clean and impute the missing data: median
> lanes per road class, and a **spatial k-nearest-neighbours** model for missing
> speed limits. A custom Haversine *point-on-segment* algorithm matches each
> sensor to its street."

🖥️ *Scroll the 4-step methodology row.*

> "We bucket traffic into four congestion levels, confirm with **ANOVA and Tukey
> tests** that road class and speed genuinely separate them, and benchmark five
> models — from a baseline up to a **decision tree, Random Forest, and finally
> XGBoost**, which wins. That's the bagging-versus-boosting story from Topic 2."

🖥️ *Point at the five metric cards, then the feature-importance chart.*

> "And we evaluate it honestly, with the Topic 1 metrics: **F1-macro 0.53**,
> **Cohen's kappa 0.39**, a **macro AUC of 0.77** — about **5× a majority
> baseline**, from *only the static shape of the street*. The confusion matrix
> shows errors fall between *adjacent* levels, the ROC curves show peak
> congestion is the most separable, and the **calibration curve** — with an
> expected calibration error of just **0.04** — means the model's confidence is
> trustworthy."

🖥️ *Pan across confusion matrix → ROC curves → calibration plot. Then briefly
open the "Live data feed" expander to show the 🟢 real-time Valencia feed.*

---

### 2:35 – 3:30 · Tab 2 — the scenario simulator (the benefit)
> "This is where it becomes a planning tool. Pick any street…"

🖥️ *Select a recognisable street from the dropdown; show its level + probability bars.*

> "…and ask *what if we redesigned it?* It's **live inference on the same
> XGBoost model — not hard-coded rules**. Let's add lanes and raise the speed
> limit."

🖥️ *Raise the lanes slider, bump the speed limit; the bars shift toward high/peak live.*

> "The model immediately re-estimates congestion. A planner can weigh that
> trade-off **before** spending a single euro — and without installing a sensor."

---

### 3:30 – 4:20 · Tab 3 — transfer to a sensorless town
> "Finally, the part we're proudest of. Paiporta, next to Valencia, has **zero**
> traffic sensors. We take the model trained only on Valencia, plus the
> road-class priors it learned, and produce a **complete congestion map for
> Paiporta** — entirely from OpenStreetMap geometry."

🖥️ *Open the Transfer tab; show the Paiporta map and the "0 sensors / 100 %
coverage" metrics.*

> "Every coloured street here was estimated with no local ground truth. That's
> the real value: a method that gives *any* town its first congestion map for free."

---

### 4:20 – 4:55 · Close
> "So — open data plus a transparent ML pipeline turns 2.4 % of measured streets
> into a city-wide, *and* town-wide, congestion map, with a live what-if
> simulator for planners. It's deployed as a server-side app combining batch
> predictions with a real-time feed — exactly the deployment patterns from
> Topic 5. The code, the live app, and the data sources are all in the
> description. Thanks for watching."

🖥️ *End on the live app URL / GitHub repo on screen.*

---

## Checklist before recording
- [ ] App **deployed** and opened in a clean browser window (hide the bookmarks bar; full-screen).
- [ ] **Live tab** shows the 🟢 real-time feed (works from a normal network, not the sandbox).
- [ ] On Tab 1, the **map has finished rendering** before you start talking over it.
- [ ] Pick a **recognisable street** for the simulator (e.g. a known avenue) so the audience relates.
- [ ] Have the **Model tab pre-scrolled** so ROC + calibration are one quick pan away.
- [ ] Keep total **under 5:00** — the rubric is strict on length. Do a timed dry-run.
- [ ] Split narration across **Imad, Raúl & Nouh** (one section each) to satisfy "video of the team".

## One-liners you can drop in if you have spare seconds
- *Originality:* "We predict congestion **where there is no sensor** — and transfer it to a town that has none."
- *Difficulty:* "OSM graphs, spatial kNN imputation, a custom geo-matching algorithm, five benchmarked models, and a live deployment."
- *DS methods:* "Imputation, ANOVA/Tukey, boosting, cross-validation, ROC/AUC and calibration."
