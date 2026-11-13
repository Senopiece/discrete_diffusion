export default {
  render({ model, el }) {
    el.classList.add("queens-plot-controls");
    // Scope styles to this widget view.
    const style = document.createElement("style");
    style.textContent = model.get("stylesheet");
    el.append(style);
    const buttons = [];
    const select = (group, values) => {
      model.set("selected", { ...model.get("selected"), [group.id]: values });
      model.save_changes();
    };
    for (const group of model.get("groups")) {
      const section = document.createElement("section");
      section.dataset.group = group.id;
      section.setAttribute("aria-label", group.label);
      const heading = document.createElement("div");
      heading.className = "queens-plot-heading";
      const label = document.createElement("strong");
      label.textContent = group.label;
      heading.append(label);
      if (group.multiple) {
        for (const [title, values] of [["All", group.options], ["None", []]]) {
          const action = document.createElement("button");
          action.type = "button";
          action.textContent = title;
          action.setAttribute("aria-label", `${title} ${group.label.toLowerCase()}`);
          action.onclick = () => select(group, values);
          heading.append(action);
        }
      }
      section.append(heading);
      const choices = document.createElement("div");
      choices.className = "queens-plot-choices";
      for (const [index, option] of group.options.entries()) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "queens-legend-item";
        const sample = group.samples?.[index];
        if (sample) {
          const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
          svg.setAttribute("viewBox", "0 0 42 16");
          svg.setAttribute("width", "42");
          svg.setAttribute("height", "16");
          svg.setAttribute("aria-hidden", "true");
          const line = document.createElementNS(svg.namespaceURI, "line");
          for (const [key, value] of Object.entries({
            x1: "2", x2: "40", y1: "8", y2: "8",
            stroke: sample.color, "stroke-width": "2",
            "stroke-dasharray": sample.dash || "none",
          })) line.setAttribute(key, value);
          svg.append(line);
          button.append(svg);
        }
        const text = document.createElement("span");
        text.textContent = option;
        button.append(text);
        button.onclick = () => {
          const current = model.get("selected")[group.id] || [];
          select(group, group.multiple
            ? (current.includes(option) ? current.filter(x => x !== option) : [...current, option])
            : [option]);
        };
        buttons.push({ button, group: group.id, option });
        choices.append(button);
      }
      section.append(choices);
      el.append(section);
    }
    const update = () => {
      const selected = model.get("selected");
      for (const { button, group, option } of buttons) {
        button.setAttribute("aria-pressed", String((selected[group] || []).includes(option)));
      }
    };
    model.on("change:selected", update);
    update();
    let chart = null;
    const onChartWheel = event => {
      // Plotly handles zoom on its inner drag layer first. Keep the same
      // wheel event from scrolling the notebook when it bubbles out.
      if (event.target.closest?.(".js-plotly-plot")) {
        event.preventDefault();
        event.stopPropagation();
      }
    };
    let lastWidth = 0;
    let frame = 0;
    const measure = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const current = el.closest(".queens-comparison")?.firstElementChild;
        if (!current) return;
        if (current !== chart) {
          if (chart) {
            observer.unobserve(chart);
            chart.removeEventListener("wheel", onChartWheel);
          }
          chart = current;
          chart.addEventListener("wheel", onChartWheel, { passive: false });
          observer.observe(chart);
        }
        const width = chart.clientWidth;
        if (width >= 10 && width !== lastWidth) {
          lastWidth = width;
          model.send({ type: "chart_resize", width });
        }
      });
    };
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    measure();
    return () => {
      observer.disconnect();
      if (chart) chart.removeEventListener("wheel", onChartWheel);
      cancelAnimationFrame(frame);
      model.off("change:selected", update);
      el.replaceChildren();
    };
  }
};
