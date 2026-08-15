import {
  BrainCircuit,
  Code2,
  Leaf,
  Server,
} from "lucide-react";

function AboutPage() {
  return (
    <div className="space-y-8">
      <section>
        <p className="text-sm font-semibold text-emerald-600">
          ABOUT
        </p>

        <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-950">
          About the Project
        </h1>

        <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-500">
          A practical machine-learning project that
          demonstrates how a trained model can become
          a usable software application.
        </p>
      </section>

      <section className="grid gap-5 md:grid-cols-2">
        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <BrainCircuit
            className="text-emerald-600"
            size={25}
          />

          <h2 className="mt-5 font-bold text-slate-950">
            Machine Learning
          </h2>

          <p className="mt-2 text-sm leading-6 text-slate-500">
            The application uses Multiple Linear
            Regression trained on vehicle fuel
            consumption data.
          </p>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <Server
            className="text-emerald-600"
            size={25}
          />

          <h2 className="mt-5 font-bold text-slate-950">
            Backend API
          </h2>

          <p className="mt-2 text-sm leading-6 text-slate-500">
            FastAPI exposes health and prediction
            endpoints that allow the frontend to
            communicate with the trained model.
          </p>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <Code2
            className="text-emerald-600"
            size={25}
          />

          <h2 className="mt-5 font-bold text-slate-950">
            Frontend
          </h2>

          <p className="mt-2 text-sm leading-6 text-slate-500">
            React, TypeScript and Tailwind CSS provide
            the interactive user interface.
          </p>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <Leaf
            className="text-emerald-600"
            size={25}
          />

          <h2 className="mt-5 font-bold text-slate-950">
            Purpose
          </h2>

          <p className="mt-2 text-sm leading-6 text-slate-500">
            The project demonstrates the complete path
            from dataset and model training to API
            integration and user-facing application.
          </p>
        </div>
      </section>
    </div>
  );
}

export default AboutPage;