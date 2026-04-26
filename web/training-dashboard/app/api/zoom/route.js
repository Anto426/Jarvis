import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

const rootDir = path.resolve(process.cwd(), "..", "..");
const zoomPath = path.join(rootDir, "logs", "zoom_state.json");

export const dynamic = "force-dynamic";

export async function POST(request) {
  try {
    const data = await request.json();
    
    let currentState = {};
    try {
      const raw = await readFile(zoomPath, "utf8");
      currentState = JSON.parse(raw);
    } catch (e) {
      // file might not exist or be invalid, start fresh
    }

    const newState = { ...currentState, ...data };
    await writeFile(zoomPath, JSON.stringify(newState, null, 2), "utf8");

    return Response.json({ success: true, state: newState }, {
      headers: { "Cache-Control": "no-store" }
    });
  } catch (error) {
    return Response.json(
      { success: false, error: error.message },
      { status: 500, headers: { "Cache-Control": "no-store" } }
    );
  }
}
