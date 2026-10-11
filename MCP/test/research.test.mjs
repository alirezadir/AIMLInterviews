import assert from "node:assert/strict";
import { test } from "node:test";
import { research, SEARCH_URL, USER_AGENT } from "../examples/research.mjs";

function remoteFixture({ toolError = false } = {}) {
  const requests = [];
  const fetch = async (url, init) => {
    assert.equal(String(url), SEARCH_URL);
    const headers = new Headers(init.headers);
    assert.equal(headers.get("user-agent"), USER_AGENT);
    assert.equal(headers.has("authorization"), false);
    if (!init.body) return new Response(null, { status: 405 });
    const request = JSON.parse(init.body);
    requests.push(request);
    if (request.id === undefined) return new Response(null, { status: 202 });
    let result;
    switch (request.method) {
      case "initialize":
        result = { protocolVersion: "2025-03-26", capabilities: { tools: {} }, serverInfo: { name: "fixture", version: "1" } };
        break;
      case "tools/list":
        result = { tools: ["web_search", "web_fetch"].map((name) => ({ name, inputSchema: { type: "object" } })) };
        break;
      case "tools/call":
        result = { content: [{ type: "text", text: toolError ? "Search unavailable" : "RAG learning resource https://example.org/rag" }], ...(toolError ? { isError: true } : {}) };
        break;
      default:
        assert.fail(`Unexpected MCP method ${request.method}`);
    }
    return Response.json({ jsonrpc: "2.0", id: request.id, result });
  };
  return { fetch, requests };
}

test("research uses the stdio tutor and anonymous HTTP search/fetch with one session", async () => {
  const fixture = remoteFixture();
  const result = await research("retrieval augmented generation", "https://example.org/rag", { fetch: fixture.fetch, output: () => {} });
  assert.ok(result.curriculum.structuredContent.returned > 0);
  assert.ok(result.curriculum.structuredContent.sections[0].source);
  assert.match(result.search.content[0].text, /RAG learning resource/);
  assert.ok(result.page);
  const calls = fixture.requests.filter((r) => r.method === "tools/call");
  assert.deepEqual(calls.map((r) => r.params.name), ["web_search", "web_fetch"]);
  const [search, page] = calls.map((r) => r.params.arguments);
  assert.deepEqual(search.search_queries, ["retrieval augmented generation machine learning explanation"]);
  assert.deepEqual(page.urls, ["https://example.org/rag"]);
  assert.equal(page.session_id, search.session_id);
  assert.match(search.session_id, /^[a-f0-9-]{36}$/);
  assert.equal(search.model_name, undefined);
  assert.ok(fixture.requests.some((r) => r.method === "tools/list"));
});

test("search-only research never fetches a page", async () => {
  const fixture = remoteFixture();
  await research("evaluation", undefined, { fetch: fixture.fetch, output: () => {} });
  assert.deepEqual(fixture.requests.filter((r) => r.method === "tools/call").map((r) => r.params.name), ["web_search"]);
});

test("remote tool failures surface instead of reporting success", async () => {
  const fixture = remoteFixture({ toolError: true });
  await assert.rejects(research("evaluation", undefined, { fetch: fixture.fetch, output: () => {} }), /web_search: Search unavailable/);
});

test("invalid arguments fail before network access", async () => {
  const fetch = () => assert.fail("Unexpected network request");
  for (const topic of [undefined, "", " ", "x".repeat(201)]) {
    await assert.rejects(research(topic, undefined, { fetch }), /Supply a topic/);
  }
  await assert.rejects(research("evaluation", "file:///tmp/answer", { fetch }), /HTTP or HTTPS/);
});
