"""Sampler comparison plots with independent temperature, budget and layer filters."""

from pathlib import Path

import anywidget
import ipywidgets as widgets
import plotly.graph_objects as go
import traitlets as tl


class PlotControls(anywidget.AnyWidget):
    groups = tl.List().tag(sync=True)
    selected = tl.Dict().tag(sync=True)
    stylesheet = tl.Unicode().tag(sync=True)

    def __init__(self, groups, selected):
        assets = Path(__file__).parent / "components" / "plot_controls"
        super().__init__(
            _esm=(assets / "plot_controls.js").read_text(encoding="utf-8"),
            stylesheet=(assets / "plot_controls.css").read_text(encoding="utf-8"),
            groups=groups, selected=selected,
        )
        self.layout.margin = "0"
        self.layout.width = "100%"


class SamplingComparison(widgets.GridBox):
    """A chart with independent temperature, budget and layer selectors."""

    def __init__(self, figure, groups, selected, trace_keys):
        self.controls = PlotControls(groups, selected)
        self.chart = go.FigureWidget(figure)
        self.chart._config = {**self.chart._config, "responsive": True,
                               "displaylogo": False, "scrollZoom": True}
        self.controls.layout.max_height = f"{figure.layout.height}px"
        self.controls.layout.overflow = "hidden"
        self._trace_keys = trace_keys
        super().__init__(
            [self.chart, self.controls],
            layout=widgets.Layout(
                width="100%", min_width="0", margin="0", padding="0", border="0",
                grid_template_columns="minmax(0, 1fr) minmax(210px, 26%)",
                align_items="flex-start",
            ),
        )
        self.add_class("queens-comparison")
        self.controls.observe(self._selection_changed, names="selected")
        self.controls.on_msg(self._resize_chart)
        self._selection_connected = True
        self._selection_changed()

    def _resize_chart(self, widget, event, buffers):
        width = event.get("width")
        if event.get("type") == "chart_resize" and isinstance(width, int) and width >= 10:
            self.chart.update_layout(width=width, autosize=False)

    def _selection_changed(self, change=None):
        selected = self.controls.selected
        with self.chart.batch_update():
            for trace, (tau, budget, layer) in zip(self.chart.data, self._trace_keys, strict=True):
                trace.visible = (tau in selected.get("tau", [])
                                 and budget in selected.get("budget", [])
                                 and layer in selected.get("layers", []))

    def close(self):
        if getattr(self, "_selection_connected", False):
            self._selection_connected = False
            self.controls.unobserve(self._selection_changed, names="selected")
            self.controls.on_msg(self._resize_chart, remove=True)
            self.controls.close()
            self.chart.close()
        super().close()


def confidence_retry_plot(statistics, taus, budgets, *, previous=None,
                          budget_label="Generation steps", budget_name="steps"):
    """Color encodes tau; dash encodes the budget. Controls filter their product."""
    tau_names = [f"{tau:g}" for tau in taus]
    budget_names = [f"{budget:g}" for budget in budgets]
    colors = ("#75A7FF", "#F4B66A", "#6DD3AC")
    dashes = ("solid", "dot", "dash", "longdash", "dashdot", "longdashdot",
              "12px,3px,2px,3px,2px,3px")
    svg_dashes = ("", "2,4", "8,6", "12,6", "8,4,2,4", "12,4,2,4", "12,3,2,3,2,3")
    figure = go.Figure()
    keys = []
    for i, tau in enumerate(taus):
        color = colors[i % len(colors)]
        rgb = tuple(int(color[p:p + 2], 16) for p in (1, 3, 5))
        for j, budget in enumerate(budgets):
            for bound, fill in (("lower", False), ("upper", True)):
                figure.add_scatter(
                    x=statistics["attempts"], y=100 * statistics[bound][i, j],
                    mode="lines", line=dict(width=0),
                    fill="tonexty" if fill else None,
                    fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.07)",
                    hoverinfo="skip", showlegend=False,
                )
                keys.append((tau_names[i], budget_names[j], "95% interval"))
            figure.add_scatter(
                x=statistics["attempts"], y=100 * statistics["mean"][i, j],
                name=f"τ = {tau:g} · {budget_name} = {budget:g}", mode="lines",
                line=dict(color=color, width=2.5, dash=dashes[j % len(dashes)]),
                hovertemplate="%{x} attempts · %{y:.2f}%<extra>%{fullData.name}</extra>",
            )
            keys.append((tau_names[i], budget_names[j], "Solve probability"))
    figure.update_layout(
        template="plotly_dark", paper_bgcolor="#1f1f1f", plot_bgcolor="#1f1f1f",
        title=dict(text="At least one solution vs attempts", x=0.02),
        font=dict(family="Arial, sans-serif", size=16), autosize=True, height=650,
        margin=dict(l=80, r=25, t=60, b=75), showlegend=False,
        hovermode="closest", dragmode="zoom",
        xaxis=dict(title="Number of attempts", range=[0, int(statistics["attempts"][-1])],
                   gridcolor="rgba(148,163,184,0.12)", zeroline=False),
        yaxis=dict(title="At least one solution (%)", range=[0, 100],
                   gridcolor="rgba(148,163,184,0.12)", zeroline=False),
    )
    groups = [
        dict(id="tau", label="Temperature τ", options=tau_names, multiple=True,
             samples=[dict(color=colors[i % len(colors)]) for i in range(len(taus))]),
        dict(id="budget", label=budget_label, options=budget_names, multiple=True,
             samples=[dict(color="#cccccc", dash=svg_dashes[i % len(svg_dashes)])
                      for i in range(len(budgets))]),
        dict(id="layers", label="Layers", options=["Solve probability", "95% interval"], multiple=True),
    ]
    view = SamplingComparison(
        figure, groups, dict(tau=tau_names, budget=budget_names, layers=["Solve probability"]), keys,
    )
    if previous is not None:
        previous.close()
    return view
