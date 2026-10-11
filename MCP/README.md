# AIMLInterviews MCP Server

`aimlinterviews-mcp` turns MCP-compatible AI assistants into interview coaches backed by the local [AIMLInterviews](https://github.com/alirezadir/AIMLInterviews) curriculum.

It discovers practice problems, provides progressive hints, reviews candidate answers, and builds role- or company-focused preparation plans. The tutor follows a no-spoilers policy: problem retrieval excludes reference answers, hints provide progressively deeper process guidance, and answer review never executes submitted code.

## Requirements

- Node.js 20 or newer
- A local clone of the AIMLInterviews repository
- An MCP-compatible client such as Claude Code or Codex

## Quick start

Clone the repository first:

```bash
git clone https://github.com/alirezadir/AIMLInterviews.git
cd AIMLInterviews
```

Then connect the tutor from the repository directory.

### Claude Code

```bash
claude mcp add aimlinterviews -- npx -y aimlinterviews-mcp
```

### Codex

```bash
codex mcp add aimlinterviews -- npx -y aimlinterviews-mcp
```

The server searches its working directory and parent directories for the clone. For clients launched elsewhere, configure the repository path explicitly:

```json
{
  "mcpServers": {
    "aimlinterviews": {
      "command": "npx",
      "args": ["-y", "aimlinterviews-mcp"],
      "env": {
        "AIMLINTERVIEWS_ROOT": "/path/to/your/AIMLInterviews/clone"
      }
    }
  }
}
```

`AIMLINTERVIEWS_ROOT` must point to the repository root that contains `README.md` and `src/`.

## Optional web research companion

Use the [research example](examples/research.mjs) to find curriculum sections and related web reading with [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp). It connects to the local tutor over stdio and to `https://search.parallel.ai/mcp` over Streamable HTTP. Anonymous search uses Fast mode and needs no account or API key; the free endpoint has rate limits.

From your repository clone:

```bash
cd MCP
npm ci
npm run build
npm run research -- "retrieval augmented generation"
```

To extract a specific reading page as well, pass its URL:

```bash
npm run research -- "retrieval augmented generation" "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
```

The example prints source-linked curriculum matches, web search results and, when requested, page excerpts as JSON. It calls MCP tools directly and does not run an LLM or an agent loop. Only the supplied topic and optional URL go to Parallel; it does not send candidate answers or local curriculum text. External pages can contain spoilers or untrusted instructions, so use them for background study rather than during a no-spoilers mock interview. The local tutor and its default client configuration stay unchanged.

## Tools

| Tool | Purpose |
| --- | --- |
| `get_server_status` | Show the repository root, source commit, and catalog counts |
| `list_problems` | Filter practice problems by area, difficulty, tag, company mention, or query |
| `get_problem` | Retrieve a prompt and source metadata without its answer |
| `get_hint` | Request spoiler-safe hints at levels 1, 2, and 3 |
| `search_curriculum` | Find relevant chapter sections and source locations |
| `get_learning_path` | Build a role- and experience-based study sequence |
| `get_company_prep` | Build a role-based company preparation plan |
| `review_answer` | Review reasoning against an interview rubric without executing code |

The server also provides the `interview_tutor`, `mock_interview`, and `study_plan` prompts, plus `aimlinterviews://catalog` and `aimlinterviews://tutor-policy` resources.

## Content and safety model

- The server reads public Markdown under `src/`; it does not bundle a duplicate question bank.
- Stable catalog IDs are derived from the source path and title.
- Every problem and section includes its source path, line, and Git commit.
- Repository reads are path-confined, size-limited, and skip symbolic links.
- Company plans are curriculum-based role recommendations, not claims about private or current company interview questions.
- Candidate code is treated as text and is never executed.
- MCP protocol messages use stdout; diagnostics use stderr.

Restart the MCP client after pulling curriculum updates so the server rebuilds its in-memory catalog.

## Development

```bash
cd MCP
npm install
npm test
npm pack --dry-run
```

Run the compiled server directly:

```bash
AIMLINTERVIEWS_ROOT="$(cd .. && pwd)" npm start
```

Inspect it interactively:

```bash
npm run inspect
```

Publishing requires npm access to the `aimlinterviews-mcp` package:

```bash
npm login
npm publish --access public
```

## Design reference

The tool categories and tutor workflow were informed by the [TorchLeet MCP server](https://github.com/Exorust/TorchLeet/tree/main/mcp-server). This implementation is original and reads the AIMLInterviews curriculum rather than copying TorchLeet content.
