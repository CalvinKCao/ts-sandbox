---
theme: seriph
background: https://images.unsplash.com/photo-1451187580459-43490279c0fa?ixlib=rb-1.2.1&auto=format&fit=crop&w=1920&q=80
class: text-center
highlighter: shiki
lineNumbers: false
info: |
  Diffusion TSF Final Presentation.
  Restoring signal texture.
drawings:
  persist: false
transition: fade
title: Diffusion-TSF
---

<h1>Diffusion TSF</h1>
<h2>Restoring Signal Texture to Forecasting</h2>

Final Presentation

<div class="pt-12">
  <span class="opacity-70 text-sm font-serif">Capturing high frequency geometric structures.</span>
</div>

<style>
h1 {
  @apply text-6xl font-serif font-bold text-white drop-shadow-lg;
}
h2 {
  @apply text-2xl font-sans font-light text-slate-200 mt-4 tracking-wide;
}
.slidev-layout.cover {
  background: linear-gradient(rgba(2, 6, 23, 0.9), rgba(2, 6, 23, 0.9)), url('https://images.unsplash.com/photo-1451187580459-43490279c0fa?ixlib=rb-1.2.1&auto=format&fit=crop&w=1920&q=80');
  background-size: cover;
}
</style>

---
layout: center
class: text-center
---

<h1>Acknowledgements</h1>

<div class="flex justify-center gap-20 mt-10">
  <div>
    <h3 class="text-2xl font-serif font-bold text-slate-800 border-b-2 border-slate-100 pb-2">Boyu Wang</h3>
    <p class="opacity-70 text-sm font-sans uppercase tracking-widest mt-2">Supervisor</p>
  </div>
  <div>
    <h3 class="text-2xl font-serif font-bold text-slate-800 border-b-2 border-slate-100 pb-2">Zhimin Mei</h3>
    <p class="opacity-70 text-sm font-sans uppercase tracking-widest mt-2">PhD Co-supervisor</p>
  </div>
</div>

<style>
h1 { @apply font-serif text-slate-900 border-b-2 border-slate-200 pb-2 mb-8; }
</style>

---

<h1>Foundations of the Study</h1>

Core concepts in signal analysis and prediction.

<div class="grid grid-cols-2 gap-8 mt-10">
<div class="space-y-6 text-slate-700">

<h3 class="font-serif text-xl text-slate-800 font-bold">Signal Forecasting</h3>

1. Time series consist of data points ordered in time.
2. Forecasting uses past observations to predict future values.
3. Accurate predictions inform operational safety and planning.

<img src="/ts_forecasting_intro.png" class="h-32 rounded shadow-sm" alt="Time Series Forecasting Visualization" />
</div>

<div class="space-y-6 text-slate-700">

<h3 class="font-serif text-xl text-slate-800 font-bold">Deep Learning</h3>

1. Deep learning uses multi layered neural networks.
2. These systems learn complex mappings from input to output.
3. Large datasets enable the discovery of hidden patterns.

<img src="/dl_intro.png" class="h-32 rounded shadow-sm mx-auto" alt="Deep Learning Visualization" />
</div>
</div>

<style>
h1 { @apply font-serif text-slate-900 border-b-2 border-slate-200 pb-2; }
</style>

---

<h1>Oversmoothing in Signal Forecasting</h1>

Neural networks acting as filters.

<div class="grid grid-cols-2 gap-10 mt-10">
<div class="bg-slate-50 p-6 rounded-xl shadow-sm border border-slate-100">

<h3 class="font-serif text-xl text-slate-800 mb-4 font-bold">Standard Objectives</h3>

1. Typical models minimize mean squared error.
2. Uncertain futures force the model to predict an average.
3. Smooth lines result.
4. Geometric structures disappear.
</div>

<div class="border-l-4 border-slate-300 pl-10">

<h3 class="font-serif text-xl text-slate-800 mb-4 font-bold">Spectral Bias</h3>

1. Systems learn broad patterns first.
2. High frequency details are ignored.
3. Deep networks function as low pass filters.
4. Sharp edges are sanded down.
</div>
</div>

<style>
h1 { @apply font-serif text-slate-900 border-b-2 border-slate-200 pb-2; }
</style>

---

<h1>Time Series Textures</h1>

Geometric primitives like sharp steps or spikes.

<div class="grid grid-cols-2 gap-6 items-center">
<div class="space-y-6 text-slate-700">

1. High frequency structure defines signal character.
2. Smooth spikes hide medical diagnoses in ECG.
3. Blurred surges cause missed grid failures in energy.
4. The system restores these sharp transitions.

</div>
<div class="relative">
  <img src="/diagram.png" class="rounded-xl shadow-2xl border-4 border-white" alt="Signal Texture Comparison" />
  <div class="absolute -bottom-4 -right-4 bg-slate-900 text-white px-3 py-1 text-xs rounded shadow">Figure 1. Texture vs Blur</div>
</div>
</div>

<style>
h1 { @apply font-serif text-slate-900 border-b-2 border-slate-200 pb-2 mb-10; }
</style>

---

<h1>Double Penalty Constraints</h1>

Mathematical bias against precise timing.

<div class="grid grid-cols-2 gap-10 mt-10">
<div class="text-slate-700">
  <p class="font-sans leading-relaxed">
    A sharp spike predicted late creates two errors. The model misses the actual event and predicts a false alarm later.
  </p>
  <img src="/double_penalty.png" class="mt-4 rounded shadow-sm" alt="Double Penalty Visualization" />
</div>
<div class="bg-amber-50 p-6 rounded-xl border border-amber-200 shadow-sm">
  <p class="font-serif text-slate-800">
    Optimizers encourage models to be vaguely wrong. Smooth predictions lower total error scores. This prevents the generation of sharp, useful forecasts.
  </p>
</div>
</div>

<style>
h1 { @apply font-serif text-slate-900 border-b-2 border-slate-200 pb-2 mb-10; }
</style>

---
layout: center
---

<h1>Generative Refinement</h1>

Generating a single likely reality.

<div class="grid grid-cols-2 gap-6 mt-12 text-center text-slate-800">
  <div class="p-6 bg-slate-50 rounded-lg border border-slate-100 shadow-sm">
    <h4 class="font-serif font-bold text-slate-900">Stage 1. Trend</h4>
    <p class="text-xs mt-2 uppercase tracking-widest text-slate-400">Global Trajectory</p>
  </div>
  <div class="p-6 bg-teal-50 rounded-lg border border-teal-100 shadow-sm">
    <h4 class="font-serif font-bold text-teal-900">Stage 2. Texture</h4>
    <p class="text-xs mt-2 uppercase tracking-widest text-teal-600">Local Detail</p>
  </div>
</div>

<img src="/refinement_stages.png" class="mt-8 mx-auto h-32 rounded shadow-sm" alt="Refinement Stages" />

<div class="mt-8 p-6 border-t border-slate-100 text-center text-slate-600 font-serif leading-relaxed">
Trend prediction is decoupled from texture generation. This avoids the mean smoothing trap.
</div>

<style>
h1 { @apply font-serif text-slate-900 border-b-2 border-slate-200 pb-2; }
</style>

---

<h1>Two Dimensional Encoding</h1>

Converting data into visual patterns.

<div class="grid grid-cols-2 gap-10 items-center mt-6">
<div class="space-y-6 text-slate-700">

1. Convolutional networks recognize geometric shapes.
2. Occupancy maps turn lines into solid objects.
3. The model reconstructs these visual objects.
4. 2D representations encode patterns better than 1D.

</div>
<img src="/occupancy_map.png" class="rounded shadow-xl border border-slate-100" alt="Occupancy Map Encoding" />
</div>

<style>
h1 { @apply font-serif text-slate-900 border-b-2 border-slate-200 pb-2; }
</style>

---
layout: center
---

<h1>The Hybrid Architecture</h1>

<div class="transform scale-150 py-20">
```mermaid
graph LR
  A[Past Data] --> B[iTransformer]
  B --> C[Diffusion UNet]
  C --> D[Forecast]
```
</div>

<div class="mt-20 grid grid-cols-2 gap-20 text-sm font-sans uppercase tracking-widest text-slate-500 font-bold">
<div class="text-center">Stage 1: Deterministic</div>
<div class="text-center">Stage 2: Generative</div>
</div>

<style>
h1 { @apply font-serif text-slate-900 border-b-2 border-slate-200 pb-2; }
</style>

---

<h1>Synthetic Pretraining</h1>

Learning signal physics.

<div class="grid grid-cols-2 gap-10 items-center mt-10">
<div class="space-y-4 text-slate-700">

1. Training uses 100,000 synthetic scenarios.
2. The system masters spikes and steps in controlled settings.
3. We then apply these skills to master real domains.
4. RealTS provides the artificial data.

</div>
<div class="bg-white p-4 rounded-xl shadow-inner border border-slate-100">
  <img src="/synthetic_samples.png" class="mx-auto" alt="Synthetic Sample Generation" />
  <p class="text-[10px] text-center mt-2 text-slate-400 uppercase tracking-tighter">Figure 2. Artificial scenarios</p>
</div>
</div>

<style>
h1 { @apply font-serif text-slate-900 border-b-2 border-slate-200 pb-2; }
</style>

---

<h1>Automated Parameter Search</h1>

Optimization through Optuna.

<div class="grid grid-cols-2 gap-10 mt-10">
<div class="bg-slate-50 p-6 rounded-xl shadow-sm border border-slate-100">

<h3 class="font-serif text-xl text-slate-800 mb-4 font-bold">Search Trials</h3>

1. Optuna controlled the hyperparameter search.
2. The guide model completed 20 trials.
3. The diffusion model finished 8 trials.
4. Experiments identified optimal learning rates.
</div>

<div class="text-slate-700">

<h3 class="font-serif text-xl text-slate-800 mb-4 font-bold">Data Pool</h3>

1. RealTS supplied 100,000 samples for pretraining.
2. The system trained for 200 epochs during search.
3. Early stopping based on validation loss occurred.
4. Best parameters saved for fine tuning.
</div>
</div>

<style>
h1 { @apply font-serif text-slate-900 border-b-2 border-slate-200 pb-2; }
</style>

---

<h1>Texture Results</h1>

Visual comparison of output quality.

<div class="grid grid-cols-2 gap-6 mt-10">
  <div class="space-y-4">
    <img src="/diagram.png" class="rounded shadow-md border-2 border-white" alt="Texture vs Blur" />
    <p class="text-xs text-center text-slate-500 uppercase tracking-widest">Figure 3. Restoration of geometric detail</p>
  </div>
  <div class="space-y-4">
    <img src="/comparison_ETTm2.png" class="rounded shadow-md border-2 border-white" alt="ETTm2 Success Example" />
    <p class="text-xs text-center text-slate-500 uppercase tracking-widest">Figure 4. Accuracy on ETTm2 dataset</p>
  </div>
</div>

<style>
h1 { @apply font-serif text-slate-900 border-b-2 border-slate-200 pb-2 mb-10; }
</style>

---

<h1>Accuracy Results</h1>

Restoring texture reduces error.

<div class="mt-10 overflow-hidden rounded-xl border border-slate-200 shadow-lg">
<table class="w-full text-left font-sans">
<thead class="bg-slate-900 text-white">
  <tr>
    <th class="p-4 font-serif text-lg">Dataset</th>
    <th class="p-4">iTransformer Baseline</th>
    <th class="p-4">Diffusion TSF</th>
    <th class="p-4">Improvement</th>
  </tr>
</thead>
<tbody class="divide-y divide-slate-100 bg-white text-slate-700">
  <tr>
    <td class="p-4 font-bold text-slate-900">Electricity Grid</td>
    <td class="p-4">1.258</td>
    <td class="p-4 font-bold text-slate-900">0.654</td>
    <td class="p-4 text-teal-600 font-bold">+48%</td>
  </tr>
  <tr>
    <td class="p-4 font-bold text-slate-900">Power Substations</td>
    <td class="p-4">1.161</td>
    <td class="p-4 font-bold text-slate-900">0.687</td>
    <td class="p-4 text-teal-600 font-bold">+41%</td>
  </tr>
  <tr class="bg-slate-50">
    <td class="p-4 font-bold text-slate-900">City Traffic</td>
    <td class="p-4">1.726</td>
    <td class="p-4 font-bold text-slate-900">1.381</td>
    <td class="p-4 text-teal-600 font-bold">+20%</td>
  </tr>
</tbody>
</table>
</div>

<div class="mt-10 p-4 border-l-4 border-teal-500 bg-teal-50 text-teal-900 italic font-serif">
Consistent performance gains across highly seasonal environments.
</div>

<style>
h1 { @apply font-serif text-slate-900 border-b-2 border-slate-200 pb-2; }
</style>

---

<h1>Dimensionality Constraints</h1>

Bottlenecks in scaling.

<div class="space-y-6 text-slate-700 mt-10">

1. The system handles 32 sensors effectively.
2. Errors increase when sensor counts exceed this limit.
3. Wide image canvases create high memory usage.
4. We found these bottlenecks during large scale tests.
5. Linear scaling remains difficult for 2D representations.

</div>

<style>
h1 { @apply font-serif text-slate-900 border-b-2 border-slate-200 pb-2 mb-10; }
li { @apply mb-6 text-xl text-slate-700; }
</style>

---

<h1>Future Directions</h1>

Next steps for scaling.

<div class="grid grid-cols-2 gap-8 mt-10 text-slate-700">
<div class="space-y-6">

1. Dynamic sensor grouping will reduce image width.
2. Hierarchical networks will process wider images.

</div>
<div class="space-y-6">

1. Sparse attention will lower costs for high sensor counts.
2. Latent diffusion will denoising in compressed spaces.

</div>
</div>

<style>
h1 { @apply font-serif text-slate-900 border-b-2 border-slate-200 pb-2; }
</style>

---

<h1>Selected Sources</h1>

<div class="mt-10 space-y-4 text-sm font-sans text-slate-700 leading-relaxed">
  <p>1. Nie et al 2023. A Time Series is Worth 64 Words. Foundations of modern TSF.</p>
  <p>2. Liu et al 2024. iTransformer. Global dependency modeling.</p>
  <p>3. Le Guen et al 2023. Deep Time Series Forecasting. Analysis of oversmoothing.</p>
  <p>4. Xu et al 2019. Frequency Principle. Neural network filtering properties.</p>
</div>

<style>
h1 { @apply font-serif text-slate-900 border-b-2 border-slate-200 pb-2; }
</style>

---
layout: center
class: text-center
---

<h1>Thank You</h1>

Questions?

<div class="mt-10 text-sm opacity-40 font-sans tracking-widest uppercase font-bold text-slate-900">
Diffusion TSF Final March 2026
</div>
