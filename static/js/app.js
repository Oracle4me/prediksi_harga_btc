// Chart.js Global State Default
Chart.defaults.color = '#5a729a';
Chart.defaults.font.family = "'Space Mono', monospace";
Chart.defaults.font.size = 10;
Chart.defaults.plugins.legend.position = 'top';
Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(7,11,20,.92)';
Chart.defaults.plugins.tooltip.borderColor = '#1e2d4a';
Chart.defaults.plugins.tooltip.borderWidth = 1;
Chart.defaults.plugins.tooltip.padding = 10;

const COLORS = {
    lstm: '#f7c948',
    srnn: '#de4ea9',
    rf: '#3ecf8e',
    xgb: '#e05b8a',
    actual: '#4f8cff',
    forecast: '#a855f7'
};

const MODEL_COLORS = {
    'LSTM': COLORS.lstm,
    'Simple RNN': COLORS.srnn,
    'Random Forest': COLORS.rf,
    'XGBoost': COLORS.xgb
};

let charts = {};
let currentTab = 'lstm';
let currentForecastTab = 'overlay';

const FORECAST_ENDPOINTS = {
    overlay: {
        history: "/api/btc_history_daily",
        forecast: "/api/btc_forecast"
    },

    forecast: {
        history: null,
        forecast: "/api/btc_forecast"
    },

    daily: {
        history: "/api/btc_history_daily",
        forecast: null
    },

    weekly: {
        history: "/api/btc_history_weekly",
        forecast: null
    }
};

const FX_RATE = {
    usdToIdr: null,
    source: null,
    lastUpdate: null
};


// Helper Functions
function safeNumber(v, fallback = 0) {
    const n = Number(v);
    return Number.isFinite(n) ? n : fallback;
}

function rupiah(v) {
    const n = safeNumber(v, 0);

    if (!FX_RATE.usdToIdr) {
        return '$ ' + n.toLocaleString('en-US', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        });
    }

    const idrValue = n * FX_RATE.usdToIdr;

    return 'Rp ' + Math.round(idrValue).toLocaleString('id-ID');
}

function percent(v) {
    return safeNumber(v, 0).toFixed(2) + '%';
}

function r2Format(v) {
    return safeNumber(v, 0).toFixed(4);
}

function destroyChart(id) {
    if (charts[id]) {
        charts[id].destroy();
        delete charts[id];
    }
}

function showPredPlaceholder(message = 'Data prediksi belum tersedia. Jalankan training terlebih dahulu.') {
    destroyChart('chartPred');

    const canvas = document.getElementById('chartPred');
    if (!canvas) return;

    const parent = canvas.parentElement;

    let placeholder = document.getElementById('predPlaceholder');

    if (!placeholder) {
        placeholder = document.createElement('div');
        placeholder.id = 'predPlaceholder';
        placeholder.className = 'placeholder';
        parent.insertBefore(placeholder, canvas);
    }

    placeholder.innerHTML = `<span>${message}</span>`;
    placeholder.style.display = 'flex';
    canvas.style.display = 'none';
}

function showPredCanvas() {
    const canvas = document.getElementById('chartPred');
    const placeholder = document.getElementById('predPlaceholder');

    if (placeholder) {
        placeholder.style.display = 'none';
    }

    if (canvas) {
        canvas.style.display = 'block';
    }
}

function alignSeriesByDates(sourceDates, sourceValues, targetDates) {
    const map = new Map();

    sourceDates.forEach((date, idx) => {
        map.set(date, sourceValues[idx]);
    });

    return targetDates.map(date => {
        const value = map.get(date);
        return value === undefined ? null : value;
    });
}


// Load IDR Rate
async function loadRate() {
    try {
        const r = await fetch('/api/idr_rate?ts=' + Date.now(), {
            cache: 'no-store'
        });

        const d = await r.json();

        if (!r.ok || d.error || !d.rate) {
            throw new Error(d.error || 'Gagal mengambil kurs USD/IDR');
        }

        FX_RATE.usdToIdr = Number(d.rate);
        FX_RATE.source = d.source || 'unknown';
        FX_RATE.lastUpdate = d.last_update || null;

        console.log(`Kurs USD/IDR: ${FX_RATE.usdToIdr} | Source: ${FX_RATE.source}`);

    } catch (e) {
        console.warn('Rate error:', e);

        FX_RATE.usdToIdr = null;
        FX_RATE.source = 'unavailable';
        FX_RATE.lastUpdate = null;
    }
}


// Chart Builder
function makeLineChart(id, datasets, labels, opts = {}) {
    destroyChart(id);

    const canvas = document.getElementById(id);
    if (!canvas) return;

    const ctx = canvas.getContext('2d');

    charts[id] = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    labels: {
                        boxWidth: 10,
                        font: {
                            size: 10
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function (ctx) {
                            if (ctx.parsed.y == null) return '';
                            return ctx.dataset.label + ': ' + rupiah(ctx.parsed.y);
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(30,45,74,.6)'
                    },
                    ticks: {
                        maxTicksLimit: 8,
                        maxRotation: 0
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(30,45,74,.6)'
                    },
                    ticks: {
                        callback: value => rupiah(value)
                    }
                }
            },
            elements: {
                point: {
                    radius: 0,
                    hitRadius: 6
                },
                line: {
                    tension: .35
                }
            },
            ...opts
        }
    });
}


// Load BTC History
async function loadHistory() {
    try {
        const r = await fetch('/api/btc_history');
        const d = await r.json();

        if (!r.ok || d.error || !d.prices || !d.dates) return;

        const last = d.prices[d.prices.length - 1];
        const prev = d.prices[d.prices.length - 2];

        const btcPriceEl = document.getElementById('btcPrice');

        if (btcPriceEl) {
            btcPriceEl.textContent = rupiah(last);

            const chg = prev ? ((last - prev) / prev * 100).toFixed(2) : 0;
            btcPriceEl.className = 'stat-val ' + (chg >= 0 ? 'up' : 'down');
        }

        makeLineChart('chartHistory', [
            {
                label: 'Harga BTC',
                data: d.prices,
                borderColor: COLORS.lstm,
                backgroundColor: 'rgba(247,201,72,.07)',
                fill: true,
                borderWidth: 2
            }
        ], d.dates);

    } catch (e) {
        console.warn('History error:', e);
    }
}

// Load forecast
async function loadForecast(mode = currentForecastTab) {
    const placeholder = document.getElementById('forecastPlaceholder');
    const chartWrap = document.getElementById('forecastChartWrap');

    if (!placeholder || !chartWrap) return;

    placeholder.innerHTML = `
        <div class="spinner"></div>
        <span>Menghitung prediksi 30 hari ke depan…</span>
    `;

    placeholder.style.display = 'flex';
    chartWrap.style.display = 'none';

    try {
        const cfg = FORECAST_ENDPOINTS[mode];

        let hist = null;
        let fc = null;

        if (cfg.history) {

            const r = await fetch(cfg.history);

            if (r.ok)
                hist = await r.json();

        }

        if (cfg.forecast) {

            const r = await fetch(cfg.forecast);

            if (r.ok)
                fc = await r.json();
        }

        if (fc && fc.error) {
            placeholder.innerHTML = `<span>${fc.error}</span>`;
            placeholder.style.display = 'flex';
            chartWrap.style.display = 'none';
            return;
        }

        if (mode === "overlay") {

            if (!hist || !fc) {
                placeholder.innerHTML = '<span>Jalankan training terlebih dahulu</span>';
                return;
            }

        }

        if (mode === "forecast") {

            if (!fc) {
                placeholder.innerHTML = '<span>Forecast belum tersedia</span>';
                return;
            }

        }

        if (mode === "daily" || mode === "weekly") {

            if (!hist) {
                placeholder.innerHTML = '<span>Data historis belum tersedia</span>';
                return;
            }

        }

        const trendCard = document.getElementById('trendCard');

        if (trendCard && fc) {
            trendCard.className = `trend-card ${fc.trend_color || 'neutral'}`;

            const trendIcon = document.getElementById('trendIcon');
            const trendTitle = document.getElementById('trendTitle');
            const trendNote = document.getElementById('trendNote');

            if (trendIcon) {
                trendIcon.innerHTML = `<i class="bi ${fc.trend_icon || 'bi-dash-circle-fill'}"></i>`;
            }

            if (trendTitle) {
                trendTitle.textContent = fc.trend || 'Sideways';
            }

            if (trendNote) {
                trendNote.textContent = fc.note || '—';
            }

            trendCard.style.display = 'flex';
        } else if (trendCard) {
            trendCard.style.display = 'none';
        }

        placeholder.style.display = 'none';
        chartWrap.style.display = 'block';

        destroyChart('chartForecast');

        const canvas = document.getElementById('chartForecast');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');

        if (mode === "daily" || mode === "weekly") {

            const forecastStats = document.getElementById('forecastStats');
            if (forecastStats) {
                forecastStats.style.display = 'none';
            }

            charts['chartForecast'] = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: hist.dates,
                    datasets: [
                        {
                            label: 'Harga Aktual',
                            data: hist.prices,
                            borderColor: COLORS.actual,
                            backgroundColor: 'rgba(79,140,255,.05)',
                            fill: true,
                            borderWidth: 2,
                            pointRadius: 0,
                            tension: 0.35
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    interaction: {
                        mode: 'index',
                        intersect: false
                    },
                    plugins: {
                        legend: {
                            labels: {
                                boxWidth: 10
                            }
                        },
                        tooltip: {
                            callbacks: {
                                label: ctx => {
                                    if (ctx.parsed.y === null) return null;
                                    return ctx.dataset.label + ': ' + rupiah(ctx.parsed.y);
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            grid: { color: 'rgba(30,45,74,.6)' },
                            ticks: { maxTicksLimit: 12, maxRotation: 0 }
                        },
                        y: {
                            grid: { color: 'rgba(30,45,74,.6)' },
                            ticks: { callback: value => rupiah(value) }
                        }
                    },
                    elements: {
                        point: { radius: 0, hoverRadius: 5, hitRadius: 8 },
                        line: { tension: 0.35 }
                    }
                }
            });

            return;
        }

        if (mode === "forecast") {

            const ciUpper = fc.ci_upper;
            const ciLower = fc.ci_lower;

            charts['chartForecast'] = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: fc.dates,
                    datasets: [
                        {
                            label: 'CI Upper',
                            data: ciUpper,
                            borderColor: 'transparent',
                            backgroundColor: 'rgba(168,85,247,.12)',
                            fill: '+1',
                            pointRadius: 0,
                            tension: 0.4
                        },
                        {
                            label: 'CI Lower',
                            data: ciLower,
                            borderColor: 'transparent',
                            backgroundColor: 'rgba(168,85,247,.12)',
                            fill: false,
                            pointRadius: 0,
                            tension: 0.4
                        },
                        {
                            label: 'Prediksi LSTM 30 Hari ke Depan',
                            data: fc.forecast,
                            borderColor: COLORS.forecast,
                            borderDash: [6, 4],
                            borderWidth: 3,
                            pointRadius: 0,
                            pointHoverRadius: 5,
                            pointBackgroundColor: COLORS.forecast,
                            pointBorderColor: COLORS.forecast,
                            tension: 0.4,
                            cubicInterpolationMode: "monotone",
                            fill: false
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    interaction: {
                        mode: 'index',
                        intersect: false
                    },
                    plugins: {
                        legend: {
                            labels: {
                                boxWidth: 10,
                                filter: item => !item.text.includes('CI')
                            }
                        },
                        tooltip: {
                            callbacks: {
                                label: ctx => {
                                    if (ctx.parsed.y === null) return null;
                                    return ctx.dataset.label + ': ' + rupiah(ctx.parsed.y);
                                },
                                afterLabel: ctx => {
                                    if (ctx.datasetIndex === 2) {
                                        return `Hari ${ctx.dataIndex + 1}/30`;
                                    }
                                    return '';
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            grid: { color: 'rgba(30,45,74,.6)' },
                            ticks: { maxTicksLimit: 15, maxRotation: 0 }
                        },
                        y: {
                            grid: { color: 'rgba(30,45,74,.6)' },
                            ticks: { callback: value => rupiah(value) }
                        }
                    },
                    elements: {
                        point: { radius: 0, hoverRadius: 5, hitRadius: 8 },
                        line: { tension: 0.4 }
                    }
                }
            });

            renderForecastStats(fc, fc.last_actual_price);
            return;
        }

        const lastActualPrice = fc.last_actual_price;

        const allDates = [...hist.dates, ...fc.dates];

        const actualSeries = [
            ...hist.prices,
            ...Array(30).fill(null)
        ];

        const forecastSeries = [
            ...Array(hist.dates.length - 1).fill(null),
            hist.prices[hist.prices.length - 1],
            ...fc.forecast
        ];

        const ciUpper = [
            ...Array(hist.dates.length - 1).fill(null),
            hist.prices[hist.prices.length - 1],
            ...fc.ci_upper
        ];

        const ciLower = [
            ...Array(hist.dates.length - 1).fill(null),
            hist.prices[hist.prices.length - 1],
            ...fc.ci_lower
        ];

        charts['chartForecast'] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: allDates,
                datasets: [
                    {
                        label: 'CI Upper',
                        data: ciUpper,
                        borderColor: 'transparent',
                        backgroundColor: 'rgba(168,85,247,.12)',
                        fill: '+1',
                        pointRadius: 0,
                        tension: 0.4
                    },
                    {
                        label: 'CI Lower',
                        data: ciLower,
                        borderColor: 'transparent',
                        backgroundColor: 'rgba(168,85,247,.12)',
                        fill: false,
                        pointRadius: 0,
                        tension: 0.4
                    },
                    {
                        label: 'Harga Aktual',
                        data: actualSeries,
                        borderColor: COLORS.actual,
                        backgroundColor: 'rgba(79,140,255,.05)',
                        fill: true,
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.35
                    },
                    {
                        label: 'Prediksi LSTM 30 Hari ke Depan',
                        data: forecastSeries,
                        borderColor: COLORS.forecast,
                        borderDash: [6, 4],
                        borderWidth: 3,
                        pointRadius: 0,
                        pointHoverRadius: 5,
                        pointBackgroundColor: COLORS.forecast,
                        pointBorderColor: COLORS.forecast,
                        tension: 0.4,
                        cubicInterpolationMode: "monotone",
                        fill: false,
                        spanGaps: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: {
                        labels: {
                            boxWidth: 10,
                            filter: item => !item.text.includes('CI')
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: ctx => {
                                if (ctx.parsed.y === null) return null;
                                return ctx.dataset.label + ': ' + rupiah(ctx.parsed.y);
                            },
                            afterLabel: ctx => {
                                if (ctx.datasetIndex === 3 && ctx.dataIndex >= hist.dates.length) {
                                    const dayNum = ctx.dataIndex - hist.dates.length + 1;
                                    return `Hari ${dayNum}/30`;
                                }
                                return '';
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(30,45,74,.6)'
                        },
                        ticks: {
                            maxTicksLimit: 15,
                            maxRotation: 0
                        }
                    },
                    y: {
                        grid: {
                            color: 'rgba(30,45,74,.6)'
                        },
                        ticks: {
                            callback: value => rupiah(value)
                        }
                    }
                },
                elements: {
                    point: {
                        radius: 0,
                        hoverRadius: 5,
                        hitRadius: 8
                    },
                    line: {
                        tension: 0.4
                    }
                }
            }
        });

        renderForecastStats(fc, lastActualPrice);

    } catch (e) {
        placeholder.innerHTML = '<span>Training diperlukan sebelum prediksi</span>';
        placeholder.style.display = 'flex';
        chartWrap.style.display = 'none';

        console.warn('Forecast error:', e);
    }
}

// Render forecast stats
function renderForecastStats(fc, lastActualPrice) {
    const isUp = fc.forecast_end >= lastActualPrice;
    const change = fc.change_pct;

    const fstatEnd = document.getElementById('fstatEnd');
    const fstatPeak = document.getElementById('fstatPeak');
    const fstatLow = document.getElementById('fstatLow');
    const fstatChg = document.getElementById('fstatChg');

    if (fstatEnd) {
        fstatEnd.textContent = rupiah(fc.forecast_end);
        fstatEnd.className = 'fstat-val ' + (isUp ? 'up' : 'down');
    }

    if (fstatPeak) {
        fstatPeak.textContent = rupiah(fc.forecast_peak);
        fstatPeak.className = 'fstat-val up';
    }

    if (fstatLow) {
        fstatLow.textContent = rupiah(fc.forecast_low);
        fstatLow.className = 'fstat-val down';
    }

    if (fstatChg) {
        fstatChg.textContent = (isUp ? '+' : '') + change + '%';
        fstatChg.className = 'fstat-val ' + (isUp ? 'up' : 'down');
    }

    const forecastStats = document.getElementById('forecastStats');
    if (forecastStats) {
        forecastStats.style.display = 'grid';
    }
}

// Load Metrics
async function loadMetrics() {
    try {
        const r = await fetch('/api/metrics');
        if (!r.ok) return;

        const d = await r.json();
        const { LSTM: l, RandomForest: rf, XGBoost: xgb, SimpleRNN: srnn } = d;

        if (!l || !rf || !xgb || !srnn) return;

        const models = {
            'LSTM': l,
            'Simple RNN': srnn,
            'Random Forest': rf,
            'XGBoost': xgb
        };

        const bestR2Model = Object.entries(models)
            .reduce((best, current) =>
                current[1].R2 > best[1].R2 ? current : best
            )[0];

        const lstmMAPEEl = document.getElementById('lstmMAPE');
        if (lstmMAPEEl) {
            lstmMAPEEl.textContent = percent(l.MAPE);
        }

        const bestModel = document.getElementById('bestModel');
        if (bestModel) {
            bestModel.textContent = bestR2Model || '—';
        }

        const lstmR2El = document.getElementById('lstmR2');
        if (lstmR2El) {
            lstmR2El.textContent = r2Format(l.R2);
        }

        const lstmMAEEl = document.getElementById('lstmMAE');
        if (lstmMAEEl) {
            lstmMAEEl.textContent = rupiah(l.MAE);
        }

        // Render Metrics Table
        const rows = [l, srnn, rf, xgb].sort((a, b) => {
            return safeNumber(b.R2, 0) - safeNumber(a.R2, 0);
        });

        function cell(val, isBest, type = 'idr') {
            let s;

            if (type === 'idr') {
                s = rupiah(val);
            } else if (type === 'pct') {
                s = percent(val);
            } else if (type === 'r2') {
                s = r2Format(val);
            } else {
                s = safeNumber(val).toLocaleString('id-ID', {
                    maximumFractionDigits: 2
                });
            }

            return `<td${isBest ? " class='lstm-best'" : ""}>${s}</td>`;
        }

        let html = `
            <table class="cmp-table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>MAE</th>
                        <th>RMSE</th>
                        <th>R2</th>
                        <th>MAPE</th>
                    </tr>
                </thead>
                <tbody>
        `;

        rows.forEach((m) => {
            html += `
                <tr>
                    <td><b style="color:${MODEL_COLORS[m.model] || '#ffffff'}">${m.model}</b></td>
                    ${cell(m.MAE, false, 'idr')}
                    ${cell(m.RMSE, false, 'idr')}
                    ${cell(m.R2, m.model === bestR2Model, 'r2')}
                    ${cell(m.MAPE, false, 'pct')}
                </tr>
            `;
        });

        html += `</tbody></table>`;

        const metricsWrap = document.getElementById('metricsWrap');
        if (metricsWrap) {
            metricsWrap.innerHTML = html;
        }

        // Render LSTM Detail
        const lstmMetrics = [
            {
                label: 'R² Score',
                val: r2Format(l.R2),
                sub: 'Semakin mendekati 1, semakin baik model mengikuti data aktual.'
            },
            {
                label: 'MAPE',
                val: percent(l.MAPE),
                sub: 'Rata-rata persentase kesalahan prediksi harga.'
            },
            {
                label: 'MAE',
                val: rupiah(l.MAE),
                sub: 'Rata-rata selisih prediksi dengan harga sebenarnya.'
            },
            {
                label: 'RMSE',
                val: rupiah(l.RMSE),
                sub: 'Mengukur besar error dengan penalti lebih tinggi pada kesalahan besar.'
            },
            {
                label: 'Akurasi Prediksi',
                val: `${(100 - l.MAPE).toFixed(2)}%`,
                sub: 'Estimasi tingkat ketepatan prediksi harga.'
            }
        ];

        let detailHtml = '<div class="metric-row">';

        lstmMetrics.forEach(({ label, val, sub }) => {
            detailHtml += `
                <div class="metric-box">
                    <div class="metric-label">${label}</div>
                    <div class="metric-value">${val}</div>
                    <div class="metric-sub">${sub}</div>
                </div>
            `;
        });

        detailHtml += '</div>';

        const lstmDetail = document.getElementById('lstmDetail');
        if (lstmDetail) {
            lstmDetail.innerHTML = detailHtml;
        }

    } catch (e) {
        console.warn('Metrics error:', e);
    }
}

// Prediction Chart
async function switchTab(tab, btn) {
    currentTab = tab;

    document.querySelectorAll('.tab-btn').forEach(button => {
        button.classList.remove('active', 't-lstm', 't-rnn', 't-rf', 't-xgb');
    });

    if (btn) {
        btn.classList.add('active');
        if (tab === 'lstm') btn.classList.add('t-lstm');
        if (tab === 'rnn') btn.classList.add('t-rnn');
        if (tab === 'rf') btn.classList.add('t-rf');
        if (tab === 'xgb') btn.classList.add('t-xgb');
    }

    if (tab === 'all') {
        await loadAllPred();
    } else {
        await loadPred(tab);
    }
}

async function switchForecastTab(tab, btn) {

    currentForecastTab = tab;

    document.querySelectorAll(".forecast-tab")
        .forEach(b => b.classList.remove("active"));

    if (btn) btn.classList.add("active");

    loadForecast(tab);

}

async function loadPred(model) {
    try {
        const r = await fetch('/api/predictions/' + model);

        if (!r.ok) {
            showPredPlaceholder('Data prediksi model belum tersedia.');
            return;
        }

        const d = await r.json();

        if (d.error || !d.dates || !d.actual || !d.predicted) {
            showPredPlaceholder('Format data prediksi tidak valid.');
            return;
        }

        showPredCanvas();

        const colorMap = {
            lstm: COLORS.lstm,
            rnn: COLORS.srnn,
            rf: COLORS.rf,
            xgb: COLORS.xgb
        };

        const nameMap = {
            lstm: 'LSTM',
            rnn: 'Simple RNN',
            rf: 'Random Forest',
            xgb: 'XGBoost',
        };

        makeLineChart('chartPred', [
            {
                label: 'Harga Aktual',
                data: d.actual,
                borderColor: COLORS.actual,
                borderWidth: 2,
                backgroundColor: 'rgba(79,140,255,.05)',
                fill: true
            },
            {
                label: nameMap[model] + ' - Prediksi',
                data: d.predicted,
                borderColor: colorMap[model],
                borderDash: [4, 2],
                borderWidth: 1.5,
                fill: false
            }
        ], d.dates);

    } catch (e) {
        console.warn('loadPred error:', e);
        showPredPlaceholder('Gagal memuat grafik prediksi.');
    }
}

async function loadAllPred() {
    try {
        const r = await fetch('/api/all_predictions');

        if (!r.ok) {
            showPredPlaceholder('Data overlay belum tersedia.');
            return;
        }

        const d = await r.json();

        if (!d.lstm || !d.rf || !d.xgb || !d.rnn) {
            showPredPlaceholder('Data prediksi belum lengkap.');
            return;
        }

        if (!d.lstm.dates || !d.lstm.actual || !d.lstm.predicted) {
            showPredPlaceholder('Format data LSTM tidak valid.');
            return;
        }

        showPredCanvas();

        const labels = d.lstm.dates;

        const rfPredicted = alignSeriesByDates(
            d.rf.dates || [],
            d.rf.predicted || [],
            labels
        );

        const xgbPredicted = alignSeriesByDates(
            d.xgb.dates || [],
            d.xgb.predicted || [],
            labels
        );

        const rnnPredicted = alignSeriesByDates(
            d.rnn.dates || [],
            d.rnn.predicted || [],
            labels
        );

        makeLineChart('chartPred', [
            {
                label: 'Aktual',
                data: d.lstm.actual,
                borderColor: COLORS.actual,
                borderWidth: 2,
                backgroundColor: 'rgba(79,140,255,.05)',
                fill: true
            },
            {
                label: 'LSTM',
                data: d.lstm.predicted,
                borderColor: COLORS.lstm,
                borderWidth: 1.8,
                borderDash: [4, 2],
                fill: false
            },
            {
                label: 'Simple RNN',
                data: rnnPredicted,
                borderColor: COLORS.srnn,
                borderWidth: 1.5,
                borderDash: [5, 3],
                fill: false
            },
            {
                label: 'Random Forest',
                data: rfPredicted,
                borderColor: COLORS.rf,
                borderWidth: 1.5,
                borderDash: [6, 3],
                fill: false
            },
            {
                label: 'XGBoost',
                data: xgbPredicted,
                borderColor: COLORS.xgb,
                borderWidth: 1.5,
                borderDash: [2, 2],
                fill: false
            }
        ], labels);

    } catch (e) {
        console.warn('loadAllPred error:', e);
        showPredPlaceholder('Gagal memuat overlay prediksi.');
    }
}


// Load LSTM History
async function loadLSTMHistory() {
    const canvas = document.getElementById('chartLSTMLoss');

    if (!canvas) return;

    try {
        const r = await fetch('/api/lstm_history');
        if (!r.ok) return;

        const d = await r.json();

        if (!d.epochs || !d.loss || !d.val_loss) return;

        destroyChart('chartLSTMLoss');

        const ctx = canvas.getContext('2d');

        charts['chartLSTMLoss'] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: d.epochs,
                datasets: [
                    {
                        label: 'Train Loss',
                        data: d.loss,
                        borderColor: COLORS.lstm,
                        borderWidth: 2,
                        fill: false
                    },
                    {
                        label: 'Val Loss',
                        data: d.val_loss,
                        borderColor: '#ff6b6b',
                        borderWidth: 1.5,
                        borderDash: [4, 2],
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: {
                        labels: {
                            boxWidth: 10
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(30,45,74,.6)'
                        },
                        title: {
                            display: true,
                            text: 'Epoch',
                            color: '#5a729a'
                        }
                    },
                    y: {
                        grid: {
                            color: 'rgba(30,45,74,.6)'
                        },
                        title: {
                            display: true,
                            text: 'Loss',
                            color: '#5a729a'
                        }
                    }
                },
                elements: {
                    point: {
                        radius: 0,
                        hitRadius: 6
                    },
                    line: {
                        tension: .3
                    }
                }
            }
        });

    } catch (e) {
        console.warn('LSTM history error:', e);
    }
}


// Training SSE Log
async function startTraining() {
    const btn = document.getElementById('btnTrain');

    if (btn) {
        btn.disabled = true;
        btn.textContent = 'Training berjalan...';
    }

    const logWrap = document.getElementById('train-log-wrap');
    const log = document.getElementById('train-log');

    if (logWrap) {
        logWrap.style.display = 'block';
    }

    if (log) {
        log.textContent = '';
    }

    try {
        await fetch('/api/train', {
            method: 'POST'
        });

        const es = new EventSource('/api/train_log');

        es.onmessage = e => {
            const data = JSON.parse(e.data);

            if (data.done) {
                es.close();

                if (btn) {
                    btn.textContent = 'Training Selesai';
                    btn.disabled = false;
                }

                loadAll();

            } else {
                if (log) {
                    log.textContent += data + '\n';
                }

                if (logWrap) {
                    logWrap.scrollTop = logWrap.scrollHeight;
                }
            }
        };

        es.onerror = () => {
            es.close();

            if (btn) {
                btn.textContent = 'Coba Lagi...';
                btn.disabled = false;
            }
        };

    } catch (e) {
        console.warn('Training error:', e);

        if (btn) {
            btn.textContent = 'Coba Lagi...';
            btn.disabled = false;
        }
    }
}


// Load All
async function loadAll() {
    await Promise.all([
        loadHistory(),
        loadMetrics(),
        loadPred('lstm'),
        loadLSTMHistory(),
        loadForecast()
    ]);
}

// Init on DOMContentLoaded
window.addEventListener('DOMContentLoaded', async () => {
    await loadRate();
    await loadHistory();

    try {
        const st = await fetch('/api/status').then(r => r.json());

        if (st.trained) {
            await Promise.all([
                loadMetrics(),
                loadPred('lstm'),
                loadLSTMHistory(),
                loadForecast()
            ]);
        }
    } catch (e) {
        console.warn('Status error:', e);
    }
});