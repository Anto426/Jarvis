import { readFile } from "node:fs/promises";
import path from "node:path";

const rootDir = path.resolve(process.cwd(), "..", "..");
const metricsPath = path.join(rootDir, "logs", "training_metrics.json");

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
    return Response.json(JSON.parse(raw), {
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
