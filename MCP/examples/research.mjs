import { randomUUID } from "node:crypto";
import { fileURLToPath } from "node:url";
import { Client, StreamableHTTPClientTransport } from "@modelcontextprotocol/client";
import { StdioClientTransport } from "@modelcontextprotocol/client/stdio";

export const SEARCH_URL = "https://search.parallel.ai/mcp";
export const USER_AGENT = "AIMLInterviews-research/0.1.0";

// This is a direct MCP client example, not an LLM agent loop.
export async function research(topic, pageUrl, { fetch = globalThis.fetch, output = console.log } = {}) {
  if (typeof topic !== "string" || !topic.trim() || topic.length > 200) {
    throw new Error("Supply a topic of 1–200 characters.");
  }
  if (pageUrl && !["http:", "https:"].includes(new URL(pageUrl).protocol)) {
    throw new Error("The reading URL must use HTTP or HTTPS.");
  }

  const tutor = new Client({ name: "aimlinterviews-research", version: "0.1.0" });
  const web = new Client({ name: "aimlinterviews-research", version: "0.1.0" });
  const tutorTransport = new StdioClientTransport({
    command: process.execPath,
    args: [fileURLToPath(new URL("../dist/src/index.js", import.meta.url))],
    cwd: fileURLToPath(new URL("../../", import.meta.url)),
  });
  const webTransport = new StreamableHTTPClientTransport(new URL(SEARCH_URL), {
    fetch,
    requestInit: { headers: { "User-Agent": USER_AGENT } },
  });
  const session_id = randomUUID();
  const search_queries = [`${topic.trim()} machine learning explanation`];
  async function call(client, name, args) {
    const result = await client.callTool({ name, arguments: args }, { timeout: 60_000 });
    if (result.isError) {
      throw new Error(`${name}: ${result.content.filter((item) => item.type === "text").map((item) => item.text).join("\n")}`);
    }
    return result;
  }
  try {
    await tutor.connect(tutorTransport);
    const curriculum = await call(tutor, "search_curriculum", { query: topic, limit: 5 });
    output(JSON.stringify({ curriculum }, null, 2));
    await web.connect(webTransport);
    const { tools } = await web.listTools();
    for (const name of pageUrl ? ["web_search", "web_fetch"] : ["web_search"]) {
      if (!tools.some((tool) => tool.name === name)) throw new Error(`Parallel does not expose ${name}.`);
    }
    const search = await call(web, "web_search", {
      objective: `Find authoritative learning resources about ${topic.trim()} for AI/ML interview preparation.`,
      search_queries,
      session_id,
    });
    output(JSON.stringify({ search }, null, 2));
    let page;
    // Fetch only a page explicitly requested by the reader, not every search result.
    if (pageUrl) {
      page = await call(web, "web_fetch", {
        urls: [pageUrl], objective: `Explain ${topic.trim()}`.slice(0, 200), search_queries, session_id,
      });
      output(JSON.stringify({ page }, null, 2));
    }
    return { curriculum, search, page };
  } finally {
    await Promise.allSettled([tutor.close(), web.close()]);
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  research(process.argv[2], process.argv[3]).catch((error) => {
    console.error(`Research failed: ${error.message}`);
    process.exitCode = 1;
  });
}
