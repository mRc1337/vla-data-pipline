import { describe, expect, it } from "vitest";
import { discoverSeriesFields } from "./seriesFields";

describe("discoverSeriesFields", () => {
  it("keeps canonical LeRobot state and action fields", () => {
    expect(discoverSeriesFields({
      "observation.state": { dtype: "float32", shape: [2] },
      action: { dtype: "float32", shape: [2] },
    })).toEqual([
      { key: "observation.state", kind: "state", title: "State" },
      { key: "action", kind: "action", title: "Action" },
    ]);
  });

  it("discovers split geek_data state fields and plural actions", () => {
    expect(discoverSeriesFields({
      "observation.image.top": { dtype: "video" },
      "observation.state.joint": { dtype: "float32", shape: [12] },
      "observation.state.end": { dtype: "float32", shape: [12] },
      actions: { dtype: "float32", shape: [14] },
    })).toEqual([
      { key: "observation.state.end", kind: "state", title: "State · end" },
      { key: "observation.state.joint", kind: "state", title: "State · joint" },
      { key: "actions", kind: "action", title: "Action" },
    ]);
  });

  it("does not treat image or similarly named velocity fields as state curves", () => {
    expect(discoverSeriesFields({
      "observation.state.camera": { dtype: "video" },
      "observation.state_vel": { dtype: "float32" },
    })).toEqual([]);
  });
});
