export type SeriesFieldKind = "state" | "action";

export type SeriesField = {
  key: string;
  kind: SeriesFieldKind;
  title: string;
};

type FeatureSpec = { dtype?: unknown; shape?: unknown };

function isNumericFeature(value: unknown): boolean {
  if (!value || typeof value !== "object") return true;
  const dtype = String((value as FeatureSpec).dtype || "").toLowerCase();
  return !dtype || !["video", "image", "string", "bool", "boolean"].includes(dtype);
}

function fieldTitle(kind: SeriesFieldKind, key: string): string {
  const base = kind === "state" ? "State" : "Action";
  if (key === "observation.state" || key === "action" || key === "actions") return base;
  const prefix = kind === "state" ? "observation.state." : key.startsWith("actions.") ? "actions." : "action.";
  const suffix = key.slice(prefix.length).replace(/\./g, " / ");
  return suffix ? `${base} · ${suffix}` : base;
}

/** Discover plottable state/action columns without assuming one LeRobot naming convention. */
export function discoverSeriesFields(schema?: Record<string, unknown>): SeriesField[] {
  if (!schema) {
    return [
      { key: "observation.state", kind: "state", title: "State" },
      { key: "action", kind: "action", title: "Action" },
    ];
  }
  const keys = Object.keys(schema).filter((key) => isNumericFeature(schema[key]));
  const states = keys.filter((key) => (
    key === "observation.state" || key.startsWith("observation.state.")
  ));
  const actions = keys.filter((key) => (
    key === "action" || key === "actions"
    || key.startsWith("action.") || key.startsWith("actions.")
  ));
  const rank = (key: string): number => (
    key === "observation.state" || key === "action" ? 0 : key === "actions" ? 1 : 2
  );
  return [
    ...states.sort((left, right) => rank(left) - rank(right) || left.localeCompare(right)),
    ...actions.sort((left, right) => rank(left) - rank(right) || left.localeCompare(right)),
  ].map((key) => {
    const kind = key.startsWith("observation.state") ? "state" : "action";
    return { key, kind, title: fieldTitle(kind, key) };
  });
}
