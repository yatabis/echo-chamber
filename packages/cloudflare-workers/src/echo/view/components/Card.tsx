import type { FC, PropsWithChildren } from 'hono/jsx';

export const Card: FC<
  PropsWithChildren<{ title: string; right?: unknown }>
> = async ({ title, right, children }) => (
  <section className="card">
    <div className="row">
      <h2>{title}</h2>
      {right}
    </div>
    {children}
  </section>
);
