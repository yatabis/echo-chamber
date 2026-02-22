import type { FC, PropsWithChildren } from 'hono/jsx';

export const Layout: FC<PropsWithChildren<{ title?: string }>> = async ({
  title = 'Echo Status',
  children,
}) => (
  <html lang="ja">
    <head>
      <meta charSet="utf-8" />
      <meta
        name="viewport"
        content="width=device-width, initial-scale=1, viewport-fit=cover"
      />
      <title>{title}</title>
      <link rel="icon" href="/favicon.ico" />
      <link rel="apple-touch-icon" href="/apple-touch-icon.png" />
      <style>{`
        :root { --bg: #0b0f14; --card: #101721; --text: #e6edf3; --muted:#98a2b3; --accent:#6ee7b7; --warn:#f59e0b; --danger:#ef4444; --line:#1f2a37; }
        *{box-sizing:border-box}
        html,body{margin:0;padding:0;background:var(--bg);color:var(--text);font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Noto Sans, Helvetica Neue, Arial, Apple Color Emoji, Segoe UI Emoji;}
        a{color:inherit;text-decoration:none}
        .container{max-width:720px;margin:0 auto;padding:16px}
        .header{display:flex;align-items:center;gap:12px}
        .pill{padding:4px 10px;border-radius:999px;background:#14202e;color:var(--muted);font-size:12px;border:1px solid var(--line)}
        .grid{display:grid;gap:12px}
        @media(min-width:768px){.grid{grid-template-columns:1fr 1fr}}
        .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}
        .card h2{margin:0 0 8px 0;font-size:16px}
        .muted{color:var(--muted)}
        .row{display:flex;align-items:center;justify-content:space-between;gap:8px}
        .btn{display:inline-flex;align-items:center;justify-content:center;padding:10px 12px;border-radius:10px;border:1px solid var(--line);background:#0f1722;color:var(--text);font-weight:600}
        .btn:active{transform:translateY(1px)}
        .btn-danger{background:#1a1111;border-color:#3a1a1a;color:#ffd6d6}
        .list{display:flex;flex-direction:column;gap:8px}
        .item{display:flex;gap:8px;align-items:flex-start;justify-content:space-between;border:1px solid var(--line);padding:10px;border-radius:10px;background:#0f1520}
        .item h3{margin:0 0 4px 0;font-size:14px}
        .small{font-size:12px}
        .mono{font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace}
        .chart{height:200px;display:flex;flex-direction:column;gap:12px;border:1px dashed #223047;border-radius:10px;padding:10px}
        .chart-bars{display:flex;align-items:flex-end;gap:8px;flex:1;min-height:120px}
        .bar{flex:1;display:flex;flex-direction:column-reverse;align-items:center;gap:6px;height:100%}
        .bar-rect{width:100%;border-radius:6px;background:linear-gradient(180deg, #34d399 0%, #22c55e 100%)}
        .bar-stack{width:100%;border-radius:6px;display:flex;flex-direction:column;justify-content:flex-end;position:relative;min-height:1px}
        .bar-segment{width:100%;flex-shrink:0}
        .bar-segment:first-child{border-radius:6px 6px 0 0}
        .bar-segment:last-child{border-radius:0 0 6px 6px}
        .bar-segment:only-child{border-radius:6px}
        .bar-x{color:var(--muted);font-size:10px}
        .bar-y{color:var(--muted);font-size:11px}
        .usage-legend{display:flex;flex-wrap:wrap;gap:12px;justify-content:center;align-items:center;padding-top:8px;border-top:1px solid var(--line)}
        .legend-item{display:flex;align-items:center;gap:6px;font-size:12px}
        .legend-color{width:12px;height:12px;border-radius:2px;border:1px solid var(--line)}
        .section-title{font-size:14px;margin:0 0 8px 0}
      `}</style>
      <script
        src="https://cdn.jsdelivr.net/npm/htmx.org@2.0.6/dist/htmx.min.js"
        integrity="sha384-Akqfrbj/HpNVo8k11SXBb6TlBWmXXlYQrCSqEWmyKJe+hDm3Z/B2WVG4smwBkRVm"
        crossorigin="anonymous"
      ></script>
    </head>
    <body>
      <div className="container">{children}</div>
    </body>
  </html>
);
