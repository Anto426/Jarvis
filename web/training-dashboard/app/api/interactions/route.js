import { readFile } from "node:fs/promises";
import path from "node:path";

const rootDir = path.resolve(process.cwd(), "..", "..");
const interactionsPath = path.join(rootDir, "data", "jarvis_interactions.txt");

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const content = await readFile(interactionsPath, "utf8");
    
    // Split into films/scenes
    const sections = content.split(/# FILM: /).filter(Boolean);
    const films = sections.map(section => {
      const lines = section.split("\n");
      const title = lines[0].trim();
      const body = lines.slice(1).join("\n").trim();
      
      // Split into scenes
      const scenes = body.split("--- SCENA ESTRATTA ---").map(s => s.trim()).filter(Boolean);
      
      return {
        title,
        scenes: scenes.map(scene => {
          // Parse lines like [SPEAKER]: text
          const dialogueLines = scene.split("\n").map(l => {
            const match = l.match(/^\[([^\]]+)\]: (.*)$/);
            if (match) {
              return { speaker: match[1], text: match[2] };
            }
            return { speaker: "INFO", text: l };
          });
          return { dialogue: dialogueLines };
        })
      };
    });

    return Response.json({ films }, {
      headers: { "Cache-Control": "no-store" }
    });
  } catch (error) {
    return Response.json({ films: [], error: error.message }, { status: 500 });
  }
}
