import type { ReactNode } from "react";

interface CardProps {
  children: ReactNode;
  className?: string;
}

function Card({
  children,
  className = "",
}: CardProps) {
  return (
    <section
      className={[
        "rounded-2xl border border-slate-200",
        "bg-white shadow-sm",
        className,
      ].join(" ")}
    >
      {children}
    </section>
  );
}

export default Card;