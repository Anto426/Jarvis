import { readFile } from "node:fs/promises";
import path from "node:path";

const rootDir = path.resolve(process.cwd(), "..", "..");
const metricsPath = path.join(rootDir, "logs", "training_metrics.json");
const zoomPath = path.join(rootDir, "logs", "zoom_state.json");

const waitingPayload = {
  status: "waiting",
  updated_at: null,
  total_steps: null,
  current: {},
  history: [],
  epochs: []
};

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const raw = await readFile(metricsPath, "utf8");
    const payload = JSON.parse(raw);

    // Inject zoom state if available
    try {
      const zoomRaw = await readFile(zoomPath, "utf8");
      const zoomState = JSON.parse(zoomRaw);
      if (zoomState.lossDomain !== undefined) payload.lossDomain = zoomState.lossDomain;
      if (zoomState.lrDomain !== undefined) payload.lrDomain = zoomState.lrDomain;
    } catch (e) {
      // Zoom file might not exist yet, that's fine
    }

    return Response.json(payload, {
      headers: { "Cache-Control": "no-store" }
    });
  } catch (error) {
    if (error.code === "ENOENT") {
      return Response.json(waitingPayload, {
        headers: { "Cache-Control": "no-store" }
      });
    }

    if (error instanceof SyntaxError) {
      return Response.json(
        { status: "reading", history: [], epochs: [] },
        { status: 503, headers: { "Cache-Control": "no-store" } }
      );
    }

    return Response.json(
      { status: "error", history: [], epochs: [] },
      { status: 500, headers: { "Cache-Control": "no-store" } }
    );
  }
}
