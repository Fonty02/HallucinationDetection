import type { DesignSystem, Page, SlideMeta } from '@open-slide/core';
import { useSlidePageNumber } from '@open-slide/core';

import unibaLogo from './assets/uniba-logo.png';
import miniAttn from './assets/mini_accuracy_attn_models_by_dataset.png';
import bbcChart from './assets/GemmaToLlama_BBC_accuracy_mean.png';
import heChart from './assets/GemmaToLlama_HE_accuracy_mean.png';
import crossDomainGemmaToLlama from './assets/fig_cross_domain_gemma-2-9b-it_to_Llama-3.1-8B-Instruct.png';
import gemmaResults from './assets/gemma_results.png';
import gemmaSlimResults from './assets/gemma_slim_results.png';
import CiLabLogo from './assets/cilab.png';
import truthxDiagram from './assets/TruthX.png';
import slimTruthDiagram from './assets/SlimTruth.png';
import catImg from './assets/cat.png';

export const design: DesignSystem = {
  palette: { bg: '#ffffff', text: '#1f2933', accent: '#1f3a56' },
  fonts: {
    display: 'Georgia, "Times New Roman", serif',
    body: '"Palatino Linotype", "Book Antiqua", Palatino, "Times New Roman", serif',
  },
  typeScale: { hero: 160, body: 36 },
  radius: 8,
};

const muted = '#5b6776';
const borderColor = 'rgba(31,41,51,0.18)';
const surface = 'rgba(31,41,51,0.04)';
const chartFilter = 'saturate(0.85) contrast(1.04)';
const PX = 120;

const fill = {
  width: '100%',
  height: '100%',
  fontFamily: 'var(--osd-font-body)',
} as const;

const Footer = () => {
  const { current, total } = useSlidePageNumber();
  const showCat = current !== 1 && current !== total;
  return (
    <>
      {showCat && (
        <img src={catImg} style={{ position: 'absolute', bottom: 16, left: 16, height: 56, objectFit: 'contain', opacity: 0.9 }} />
      )}
      <div style={{
        position: 'absolute', bottom: 36, right: PX,
        fontSize: 18, color: muted, display: 'flex', alignItems: 'center', gap: 10,
      }}>
        <div style={{ width: 28, height: 1, background: 'var(--osd-accent)', opacity: 0.35 }} />
        <span>{String(current).padStart(2, '0')} / {String(total).padStart(2, '0')}</span>
      </div>
    </>
  );
};

// ============================================================
// PAGE 1 — Cover
// ============================================================
const Cover: Page = () => (
  <div
    style={{ ...fill, background: 'var(--osd-bg)', color: 'var(--osd-text)', display: 'flex', flexDirection: 'column', justifyContent: 'center', padding: `0 ${PX}px`, position: 'relative' }}
  >
    <img 
      src={CiLabLogo} 
      style={{ position: 'absolute', top: PX - 100, left: PX -100 , width: '10%', opacity: 1.0  }} 
    />
    <img
      src={unibaLogo}
      style={{ position: 'absolute', top: PX - 120, right: PX - 100, width: '8.5%', opacity: 1.0 }}
    />
    <div style={{ fontSize: 28, color: 'var(--osd-accent)', letterSpacing: '0.25em', marginBottom: 32, textAlign: 'center' }}>
      UNIVERSITY OF BARI ALDO MORO
    </div>
    <h1
      style={{ fontFamily: 'var(--osd-font-display)', fontSize: '65px', fontWeight: 900, lineHeight: 1.05, margin: 0, maxWidth: 1500 }}
    >
      Towards Cross-Model Transferability of Hallucination Detection and Mitigation in Large Language Models
    </h1>
    <p style={{
      fontFamily: 'var(--osd-font-display)',
      fontSize: 58,
      lineHeight: 1.2,
      color: muted,
      margin: '28px 0 0 0',
      maxWidth: 1300,
    }}>{''}</p>
    <div style={{
      display: 'flex', alignItems: 'center', gap: 20,
      marginTop: 56, borderTop: `1px solid ${borderColor}`, paddingTop: 32,
    }}>
      <div>
        <div style={{ fontSize: 60, fontWeight: 600 }}>Emanuele Fontana</div>
        <div style={{ fontSize: 40, color: muted }}>Department of Computer Science</div>
        <div style={{ fontSize: 35, color: muted }}>Thesis in Deep Learning</div>
        <div style={{ fontSize: 30, color: muted, marginTop: 16 }}>
          <span style={{ color: 'var(--osd-accent)' }}>Supervisor:</span>{' Dr. Gennaro Vessio · Co-supervisor:'}
          <span style={{ color: 'var(--osd-accent)' }}>{''}</span>{' Lucrezia Laraspata'}
        </div>
      </div>
    </div>
    <Footer />
  </div>
);

// ============================================================
// PAGE 2 — Background
// ============================================================
const subHeading: React.CSSProperties = {
  fontSize: 50,
  fontWeight: 700,
  color: 'var(--osd-accent)',
  letterSpacing: '0.1em',
  marginBottom: 8,
};

const TransformerDiagram = () => (
  <svg viewBox="0 0 340 282" style={{ width: '100%', height: '100%' }}>
    <defs>
      <marker id="arrowSmall" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
        <path d="M0,0 L10,5 L0,10 Z" fill="rgba(31,58,86,0.5)" />
      </marker>
    </defs>

    {/* Input tokens */}
    <text x="170" y="14" textAnchor="middle" fill={muted} fontSize="11" fontWeight="600" fontFamily="system-ui">Input tokens</text>
    <rect x="40" y="20" width="58" height="20" rx="3" fill="rgba(31,58,86,0.12)" stroke="rgba(31,58,86,0.3)" strokeWidth="1" />
    <rect x="112" y="20" width="58" height="20" rx="3" fill="rgba(31,58,86,0.12)" stroke="rgba(31,58,86,0.3)" strokeWidth="1" />
    <rect x="184" y="20" width="58" height="20" rx="3" fill="rgba(31,58,86,0.12)" stroke="rgba(31,58,86,0.3)" strokeWidth="1" />
    <rect x="256" y="20" width="44" height="20" rx="3" fill="rgba(31,58,86,0.12)" stroke="rgba(31,58,86,0.3)" strokeWidth="1" />

    <line x1="170" y1="44" x2="170" y2="62" stroke="rgba(31,58,86,0.5)" strokeWidth="1.4" markerEnd="url(#arrowSmall)" />

    {/* Embedding */}
    <rect x="24" y="64" width="292" height="28" rx="4" fill="rgba(31,58,86,0.08)" stroke="rgba(31,58,86,0.25)" strokeWidth="1" />
    <text x="170" y="82" textAnchor="middle" fill="var(--osd-text)" fontSize="11" fontWeight="600" fontFamily="system-ui">Token + Positional Embedding</text>

    <line x1="170" y1="96" x2="170" y2="114" stroke="rgba(31,58,86,0.5)" strokeWidth="1.4" markerEnd="url(#arrowSmall)" />

    {/* Transformer block (×N) — label sits inside the block, clear of the arrow */}
    <rect x="24" y="116" width="292" height="108" rx="6" fill="rgba(31,58,86,0.04)" stroke="rgba(31,58,86,0.45)" strokeWidth="1.4" />
    <text x="170" y="136" textAnchor="middle" fill="var(--osd-accent)" fontSize="12" fontWeight="700" fontFamily="system-ui">Transformer Block  ×N layers</text>
    <rect x="40" y="146" width="260" height="30" rx="4" fill="rgba(31,58,86,0.09)" stroke="rgba(31,58,86,0.25)" strokeWidth="0.9" />
    <text x="170" y="165" textAnchor="middle" fill="var(--osd-text)" fontSize="11" fontWeight="600" fontFamily="system-ui">Multi-Head Self-Attention</text>
    <rect x="40" y="184" width="260" height="30" rx="4" fill="rgba(31,58,86,0.09)" stroke="rgba(31,58,86,0.25)" strokeWidth="0.9" />
    <text x="170" y="203" textAnchor="middle" fill="var(--osd-text)" fontSize="11" fontWeight="600" fontFamily="system-ui">Feed-Forward (MLP)</text>

    <line x1="170" y1="228" x2="170" y2="246" stroke="rgba(31,58,86,0.5)" strokeWidth="1.4" markerEnd="url(#arrowSmall)" />

    {/* Output */}
    <rect x="24" y="248" width="292" height="28" rx="4" fill="rgba(31,58,86,0.08)" stroke="rgba(31,58,86,0.25)" strokeWidth="1" />
    <text x="170" y="266" textAnchor="middle" fill="var(--osd-text)" fontSize="11" fontWeight="600" fontFamily="system-ui">LayerNorm + Linear → Next token</text>
  </svg>
);

const PlatonicHypothesisDiagram = () => (
  <svg viewBox="0 0 340 300" style={{ width: '100%', height: '100%' }}>
    <defs>
      <marker id="arrowGold3" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
        <path d="M0,0 L10,5 L0,10 Z" fill="var(--osd-accent)" />
      </marker>
    </defs>

    {/* Converging arrows (drawn first, under the nodes) */}
    <line x1="92" y1="64" x2="132" y2="100" stroke="var(--osd-accent)" strokeWidth="1.4" markerEnd="url(#arrowGold3)" />
    <line x1="248" y1="64" x2="208" y2="100" stroke="var(--osd-accent)" strokeWidth="1.4" markerEnd="url(#arrowGold3)" />
    <line x1="92" y1="176" x2="132" y2="142" stroke="var(--osd-accent)" strokeWidth="1.4" markerEnd="url(#arrowGold3)" />
    <line x1="248" y1="176" x2="208" y2="142" stroke="var(--osd-accent)" strokeWidth="1.4" markerEnd="url(#arrowGold3)" />

    {/* Shared representation Z (center) */}
    <circle cx="170" cy="120" r="46" fill="rgba(31,58,86,0.12)" stroke="var(--osd-accent)" strokeWidth="2" />
    <text x="170" y="116" textAnchor="middle" fill="var(--osd-accent)" fontSize="22" fontWeight="800" fontFamily="system-ui">Z</text>
    <text x="170" y="134" textAnchor="middle" fill="var(--osd-accent)" fontSize="9.5" fontWeight="600" fontFamily="system-ui">shared reality</text>

    {/* Generic LLM nodes */}
    <rect x="28" y="40" width="76" height="34" rx="8" fill="rgba(31,58,86,0.06)" stroke="rgba(31,58,86,0.4)" strokeWidth="1.4" strokeDasharray="5 3" />
    <text x="66" y="62" textAnchor="middle" fill="var(--osd-text)" fontSize="13" fontWeight="700" fontFamily="system-ui">LLM 1</text>
    <rect x="236" y="40" width="76" height="34" rx="8" fill="rgba(31,58,86,0.06)" stroke="rgba(31,58,86,0.4)" strokeWidth="1.4" strokeDasharray="5 3" />
    <text x="274" y="62" textAnchor="middle" fill="var(--osd-text)" fontSize="13" fontWeight="700" fontFamily="system-ui">LLM 2</text>
    <rect x="28" y="166" width="76" height="34" rx="8" fill="rgba(31,58,86,0.06)" stroke="rgba(31,58,86,0.4)" strokeWidth="1.4" strokeDasharray="5 3" />
    <text x="66" y="188" textAnchor="middle" fill="var(--osd-text)" fontSize="13" fontWeight="700" fontFamily="system-ui">LLM 3</text>
    <rect x="236" y="166" width="76" height="34" rx="8" fill="rgba(31,58,86,0.06)" stroke="rgba(31,58,86,0.4)" strokeWidth="1.4" strokeDasharray="5 3" />
    <text x="274" y="188" textAnchor="middle" fill="var(--osd-text)" fontSize="13" fontWeight="700" fontFamily="system-ui">LLM 4</text>

    {/* implies more models */}
    <text x="170" y="222" textAnchor="middle" fill={muted} fontSize="11" fontStyle="italic" fontFamily="system-ui">LLM 5, LLM 6, …</text>

    {/* Annotation */}
    <text x="170" y="252" textAnchor="middle" fill={muted} fontSize="10.5" fontFamily="system-ui">Different architectures converge to a</text>
    <text x="170" y="267" textAnchor="middle" fill={muted} fontSize="10.5" fontFamily="system-ui">shared statistical model of reality</text>
    <text x="170" y="292" textAnchor="middle" fill="var(--osd-accent)" fontSize="11.5" fontWeight="700" fontFamily="system-ui">→ Tasks transfer across different models</text>
  </svg>
);

const Background: Page = () => (
  <div style={{ ...fill, background: 'var(--osd-bg)', color: 'var(--osd-text)', padding: `60px ${PX}px`, position: 'relative' }}>
    <h2 style={{ fontFamily: 'var(--osd-font-display)', fontSize: 56, fontWeight: 900, margin: 0 }}>
      How LLMs Work &amp; Why Transfer is Possible
    </h2>
    <div style={{ display: 'flex', gap: 56, marginTop: 24, flex: 1 }}>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <div style={{ fontSize: 22, fontWeight: 700, color: 'var(--osd-accent)', letterSpacing: '0.08em', marginBottom: 4 }}>
          HOW AN LLM WORKS
        </div>
        <div style={{ fontSize: 20, color: muted, lineHeight: 1.4, textAlign: 'center', maxWidth: 560, marginBottom: 10 }}>
          A deep stack of identical Transformer blocks — every layer leaves internal activations we can read.
        </div>
        <TransformerDiagram />
      </div>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <div style={{ fontSize: 22, fontWeight: 700, color: 'var(--osd-accent)', letterSpacing: '0.08em', marginBottom: 4 }}>
          PLATONIC REPRESENTATION HYPOTHESIS
        </div>
        <div style={{ fontSize: 20, color: muted, lineHeight: 1.4, textAlign: 'center', maxWidth: 560, marginBottom: 10 }}>
          Different LLMs converge to a shared representation of reality (Huh et al., 2024).
        </div>
        <PlatonicHypothesisDiagram />
      </div>
    </div>
    <Footer />
  </div>
);

// ============================================================
// PAGE 3 — Hallucination Taxonomy
// ============================================================
const sectionLabel: React.CSSProperties = {
  fontSize: 22, fontWeight: 700, color: 'var(--osd-accent)', letterSpacing: '0.12em', marginBottom: 12,
};

const TaxonomyTree = () => (
  <svg viewBox="0 0 520 250" style={{ width: '100%', height: 'auto', display: 'block' }}>
    <defs>
      <marker id="taxArrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto">
        <path d="M0,0 L10,5 L0,10 Z" fill="rgba(31,58,86,0.4)" />
      </marker>
    </defs>

    {/* connectors: root → axes */}
    <line x1="260" y1="50" x2="140" y2="96" stroke="rgba(31,58,86,0.4)" strokeWidth="1.4" markerEnd="url(#taxArrow)" />
    <line x1="260" y1="50" x2="380" y2="96" stroke="rgba(31,58,86,0.4)" strokeWidth="1.4" markerEnd="url(#taxArrow)" />
    {/* connectors: axis "by nature" → leaves */}
    <line x1="140" y1="132" x2="50"  y2="184" stroke="rgba(31,58,86,0.35)" strokeWidth="1.2" markerEnd="url(#taxArrow)" />
    <line x1="140" y1="132" x2="140" y2="184" stroke="rgba(31,58,86,0.35)" strokeWidth="1.2" markerEnd="url(#taxArrow)" />
    <line x1="140" y1="132" x2="230" y2="184" stroke="rgba(31,58,86,0.35)" strokeWidth="1.2" markerEnd="url(#taxArrow)" />
    {/* connectors: axis "by source" → leaves */}
    <line x1="380" y1="132" x2="330" y2="184" stroke="rgba(31,58,86,0.35)" strokeWidth="1.2" markerEnd="url(#taxArrow)" />
    <line x1="380" y1="132" x2="430" y2="184" stroke="rgba(31,58,86,0.35)" strokeWidth="1.2" markerEnd="url(#taxArrow)" />

    {/* root */}
    <rect x="190" y="14" width="140" height="36" rx="8" fill="rgba(31,58,86,0.14)" stroke="var(--osd-accent)" strokeWidth="2" />
    <text x="260" y="37" textAnchor="middle" fill="var(--osd-accent)" fontSize="15" fontWeight="800" fontFamily="system-ui">HALLUCINATION</text>

    {/* axes */}
    <rect x="78" y="96" width="124" height="36" rx="8" fill="rgba(31,58,86,0.07)" stroke="rgba(31,58,86,0.45)" strokeWidth="1.4" />
    <text x="140" y="119" textAnchor="middle" fill="var(--osd-text)" fontSize="13" fontWeight="700" fontFamily="system-ui">By nature</text>
    <rect x="318" y="96" width="124" height="36" rx="8" fill="rgba(31,58,86,0.07)" stroke="rgba(31,58,86,0.45)" strokeWidth="1.4" />
    <text x="380" y="119" textAnchor="middle" fill="var(--osd-text)" fontSize="13" fontWeight="700" fontFamily="system-ui">By source</text>

    {/* by nature leaves */}
    <rect x="11"  y="184" width="78" height="32" rx="6" fill="rgba(31,58,86,0.05)" stroke="rgba(31,58,86,0.35)" strokeWidth="1" />
    <text x="50"  y="204" textAnchor="middle" fill="var(--osd-text)" fontSize="12" fontWeight="700" fontFamily="system-ui">Factual</text>
    <text x="50"  y="232" textAnchor="middle" fill={muted} fontSize="8.5" fontFamily="system-ui">false vs. reality</text>
    <rect x="101" y="184" width="78" height="32" rx="6" fill="rgba(31,58,86,0.05)" stroke="rgba(31,58,86,0.35)" strokeWidth="1" />
    <text x="140" y="204" textAnchor="middle" fill="var(--osd-text)" fontSize="12" fontWeight="700" fontFamily="system-ui">Logical</text>
    <text x="140" y="232" textAnchor="middle" fill={muted} fontSize="8.5" fontFamily="system-ui">self-contradictory</text>
    <rect x="191" y="184" width="78" height="32" rx="6" fill="rgba(31,58,86,0.05)" stroke="rgba(31,58,86,0.35)" strokeWidth="1" />
    <text x="230" y="204" textAnchor="middle" fill="var(--osd-text)" fontSize="12" fontWeight="700" fontFamily="system-ui">Context</text>
    <text x="230" y="232" textAnchor="middle" fill={muted} fontSize="8.5" fontFamily="system-ui">contradicts input</text>

    {/* by source leaves */}
    <rect x="291" y="184" width="78" height="32" rx="6" fill="rgba(31,58,86,0.05)" stroke="rgba(31,58,86,0.35)" strokeWidth="1" />
    <text x="330" y="204" textAnchor="middle" fill="var(--osd-text)" fontSize="12" fontWeight="700" fontFamily="system-ui">Intrinsic</text>
    <text x="330" y="232" textAnchor="middle" fill={muted} fontSize="8.5" fontFamily="system-ui">conflicts w/ input</text>
    <rect x="391" y="184" width="78" height="32" rx="6" fill="rgba(31,58,86,0.05)" stroke="rgba(31,58,86,0.35)" strokeWidth="1" />
    <text x="430" y="204" textAnchor="middle" fill="var(--osd-text)" fontSize="12" fontWeight="700" fontFamily="system-ui">Extrinsic</text>
    <text x="430" y="232" textAnchor="middle" fill={muted} fontSize="8.5" fontFamily="system-ui">parametric knowledge</text>
  </svg>
);

const Taxonomy: Page = () => (
  <div style={{ ...fill, background: 'var(--osd-bg)', color: 'var(--osd-text)', padding: PX, position: 'relative' }}>
    <h2 style={{ fontFamily: 'var(--osd-font-display)', fontSize: 60, fontWeight: 900, margin: 0 }}>Hallucination</h2>
    <p style={{ fontSize: 26, color: muted, marginTop: 12, lineHeight: 1.45, maxWidth: 1500 }}>
      An LLM output that is fluent and syntactically correct but <span style={{ color: 'var(--osd-text)' }}>factually wrong</span> —
      a key barrier to safe deployment in healthcare, law, and education.
    </p>
    <div style={{ display: 'flex', gap: 56, marginTop: 26, alignItems: 'flex-start' }}>
      <div style={{ flex: 1.05 }}>
        <div style={sectionLabel}>TAXONOMY</div>
        <TaxonomyTree />
      </div>
      <div style={{ flex: 0.95, display: 'flex', flexDirection: 'column', gap: 18 }}>
        <div style={sectionLabel}>EXAMPLE&nbsp;·&nbsp;BeliefBank Facts</div>
        <ExampleCard tone="ok"  statement="a turtle is a reptile." answer="Yes" truth="Yes — true fact"  verdict="Non-hallucination" />
        <ExampleCard tone="bad" statement="a turkey is a herb."    answer="Yes" truth="No — false fact"  verdict="Hallucination" />
      </div>
    </div>
    <Footer />
  </div>
);

// ============================================================
// PAGE 4 — Probing & Steering
// ============================================================
const ProbingDiagram = () => (
  <svg viewBox="0 0 430 180" style={{ width: '100%', maxWidth: 680, height: 'auto', display: 'block', margin: '0 auto' }}>
    <defs>
      <marker id="probeArrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
        <path d="M0,0 L10,5 L0,10 Z" fill="rgba(31,58,86,0.65)" />
      </marker>
    </defs>

    {/* frozen transformer layer */}
    <text x="60" y="58" textAnchor="middle" fill="#3a5b8c" fontSize="11" fontWeight="700" fontFamily="system-ui">❄ frozen</text>
    <rect x="6" y="66" width="108" height="58" rx="7" fill="rgba(31,58,86,0.06)" stroke="rgba(31,58,86,0.5)" strokeWidth="1.4" />
    <text x="60" y="91" textAnchor="middle" fill="var(--osd-text)" fontSize="13" fontWeight="700" fontFamily="system-ui">Transformer</text>
    <text x="60" y="109" textAnchor="middle" fill="var(--osd-text)" fontSize="13" fontWeight="700" fontFamily="system-ui">layer ℓ</text>

    {/* read activation → probe */}
    <text x="156" y="84" textAnchor="middle" fill={muted} fontSize="12" fontFamily="system-ui">activation hℓ</text>
    <line x1="114" y1="95" x2="196" y2="95" stroke="rgba(31,58,86,0.65)" strokeWidth="1.8" strokeDasharray="5 3" markerEnd="url(#probeArrow)" />

    {/* probe */}
    <rect x="198" y="66" width="108" height="58" rx="7" fill="#fff" stroke="rgba(31,58,86,0.5)" strokeWidth="1.4" />
    <text x="252" y="91" textAnchor="middle" fill="var(--osd-text)" fontSize="13" fontWeight="700" fontFamily="system-ui">Probe</text>
    <text x="252" y="109" textAnchor="middle" fill={muted} fontSize="12" fontFamily="system-ui">(classifier)</text>

    {/* prediction */}
    <line x1="306" y1="95" x2="348" y2="95" stroke="rgba(31,58,86,0.65)" strokeWidth="1.8" markerEnd="url(#probeArrow)" />
    <circle cx="388" cy="95" r="30" fill="rgba(47,111,78,0.14)" stroke="#2f6f4e" strokeWidth="2" />
    <text x="388" y="92" textAnchor="middle" fill="#2f6f4e" fontSize="15" fontWeight="800" fontFamily="system-ui">ŷ</text>
    <text x="388" y="107" textAnchor="middle" fill="#2f6f4e" fontSize="9.5" fontFamily="system-ui">halluc.?</text>

    <text x="215" y="158" textAnchor="middle" fill={muted} fontSize="12" fontFamily="system-ui">Model stays frozen — the activation is only read</text>
  </svg>
);

const SteeringDiagram = () => (
  <svg viewBox="0 0 430 200" style={{ width: '100%', maxWidth: 680, height: 'auto', display: 'block', margin: '0 auto' }}>
    <defs>
      <marker id="steerArrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
        <path d="M0,0 L10,5 L0,10 Z" fill="rgba(31,58,86,0.65)" />
      </marker>
    </defs>

    {/* formula header */}
    <text x="215" y="26" textAnchor="middle" fill="var(--osd-accent)" fontSize="17" fontWeight="800" fontFamily="system-ui">h′ = h + α · v</text>
    <text x="215" y="44" textAnchor="middle" fill={muted} fontSize="10.5" fontFamily="system-ui">v: steering vector (same dim) · α: strength</text>

    {/* layer ℓ */}
    <rect x="6" y="78" width="96" height="52" rx="7" fill="rgba(31,58,86,0.06)" stroke="rgba(31,58,86,0.5)" strokeWidth="1.4" />
    <text x="54" y="109" textAnchor="middle" fill="var(--osd-text)" fontSize="13" fontWeight="700" fontFamily="system-ui">Layer ℓ</text>

    {/* h → ⊕ */}
    <text x="140" y="96" textAnchor="middle" fill={muted} fontSize="13" fontStyle="italic" fontFamily="system-ui">h</text>
    <line x1="102" y1="104" x2="178" y2="104" stroke="rgba(31,58,86,0.65)" strokeWidth="1.8" markerEnd="url(#steerArrow)" />

    {/* sum node */}
    <circle cx="198" cy="104" r="17" fill="rgba(31,58,86,0.1)" stroke="var(--osd-accent)" strokeWidth="1.8" />
    <text x="198" y="111" textAnchor="middle" fill="var(--osd-accent)" fontSize="20" fontWeight="800" fontFamily="system-ui">+</text>

    {/* steering vector → ⊕ */}
    <rect x="162" y="158" width="72" height="26" rx="6" fill="rgba(176,129,46,0.12)" stroke="#a9772a" strokeWidth="1.4" />
    <text x="198" y="176" textAnchor="middle" fill="#a9772a" fontSize="13" fontWeight="800" fontFamily="system-ui">α · v</text>
    <line x1="198" y1="158" x2="198" y2="123" stroke="rgba(31,58,86,0.65)" strokeWidth="1.8" markerEnd="url(#steerArrow)" />

    {/* ⊕ → layer ℓ+1 */}
    <text x="240" y="96" textAnchor="middle" fill={muted} fontSize="11.5" fontStyle="italic" fontFamily="system-ui">h + α·v</text>
    <line x1="215" y1="104" x2="262" y2="104" stroke="rgba(31,58,86,0.65)" strokeWidth="1.8" markerEnd="url(#steerArrow)" />
    <rect x="264" y="78" width="104" height="52" rx="7" fill="rgba(31,58,86,0.06)" stroke="rgba(31,58,86,0.5)" strokeWidth="1.4" />
    <text x="316" y="109" textAnchor="middle" fill="var(--osd-text)" fontSize="13" fontWeight="700" fontFamily="system-ui">Layer ℓ+1</text>
  </svg>
);

const ProbingSteering: Page = () => (
  <div style={{ ...fill, background: 'var(--osd-bg)', color: 'var(--osd-text)', padding: PX, position: 'relative' }}>
    <h2 style={{ fontFamily: 'var(--osd-font-display)', fontSize: 64, fontWeight: 900, margin: 0 }}>
      Probing &amp; Steering
    </h2>
    <div style={{ display: 'flex', gap: 48, marginTop: 36 }}>
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: 36, fontWeight: 700, marginBottom: 16, fontFamily: 'var(--osd-font-display)' }}>
          Probing
        </div>
        <ul style={{ fontSize: 28, lineHeight: 1.5, color: muted, paddingLeft: 28, margin: 0 }}>
          <li>Read a layer's internal activation</li>
          <li>Feed it to a probe to perform a task (e.g. classification)</li>
        </ul>
        <div style={{ marginTop: 22 }}>
          <ProbingDiagram />
        </div>
      </div>
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: 36, fontWeight: 700, marginBottom: 16, fontFamily: 'var(--osd-font-display)' }}>
          Steering
        </div>
        <ul style={{ fontSize: 28, lineHeight: 1.5, color: muted, paddingLeft: 28, margin: 0 }}>
          <li>Add a scaled vector to a layer's output at inference</li>
          <li>Changes behaviour — no weight changes</li>
        </ul>
        <div style={{ marginTop: 22 }}>
          <SteeringDiagram />
        </div>
      </div>
    </div>
    <div style={{
      marginTop: 32, padding: '22px 32px',
      border: '2px solid rgba(31, 58, 86, 0.35)',
      borderRadius: 'var(--osd-radius)',
      background: 'rgba(31, 58, 86, 0.08)',
      display: 'flex', alignItems: 'center', gap: 20,
    }}>
      <div style={{ fontSize: 32, fontWeight: 700, color: 'var(--osd-accent)', flexShrink: 0 }}>&#9888;</div>
      <div style={{ fontSize: 28, lineHeight: 1.45, color: muted }}>
        <strong style={{ color: 'var(--osd-text)' }}>The problem:</strong> Most probing and steering methods are
        trained on a single model. They do <em>not</em> transfer across architectures — every new model requires new training.
      </div>
    </div>
    <Footer />
  </div>
);

// ============================================================
// PAGE 5 — Introduction & Objectives
// ============================================================
const ObjectiveCard = ({ num, title, desc }: { num: string; title: string; desc: string }) => (
  <div
    style={{
      flex: 1, padding: '44px 36px',
      border: `1px solid ${borderColor}`, borderRadius: 'var(--osd-radius)',
    }}
  >
    <div style={{ fontSize: 54, fontWeight: 900, color: 'var(--osd-accent)', opacity: 0.4, fontFamily: 'var(--osd-font-display)' }}>
      {num}
    </div>
    <div style={{ fontSize: 30, fontWeight: 700, margin: '16px 0 12px 0' }}>{title}</div>
    <div style={{ fontSize: 26, lineHeight: 1.55, color: muted }}>{desc}</div>
  </div>
);

const Objectives: Page = () => (
  <div style={{ ...fill, background: 'var(--osd-bg)', color: 'var(--osd-text)', padding: PX, position: 'relative' }}>
    <h2 style={{ fontFamily: 'var(--osd-font-display)', fontSize: 72, fontWeight: 900, margin: 0 }}>
      This Work
    </h2>
    <p style={{ fontSize: 36, lineHeight: 1.5, color: muted, marginTop: 28, maxWidth: 1400 }}>
      <span style={{ fontWeight: '700' }}>{'RQ: '}</span>{'Can we build '}<span style={{ color: 'var(--osd-accent)' }}>universal</span>{' detection and steering methods that detect and mitigate hallucinations '}<span style={{ color: 'var(--osd-text)' }}>across different LLM architectures</span>?
    </p>
    <div style={{ display: 'flex', gap: 32, marginTop: 52 }}>
      <ObjectiveCard num="01" title="Save activations by probing" desc="Internal activations of LLM's on different datasets are stored" />
      <ObjectiveCard num="02" title="Universal Detector" desc="Build a hallucination detector that transfers from one model to another" />
      <ObjectiveCard num="03" title="Universal Steering" desc="Build a hallucination mitigator that transfers from one model to another" />
    </div>
    <Footer />
  </div>
);

// ============================================================
// PAGE 6 — Models, Datasets & Hallucination Rates
// ============================================================
const hrCell: React.CSSProperties = { padding: '16px 24px', fontSize: 30, textAlign: 'center', borderBottom: `1px solid ${borderColor}` };
const hrHeader: React.CSSProperties = { ...hrCell, fontSize: 20, fontWeight: 700, color: muted, letterSpacing: '0.1em' };

const ModelsDatasets: Page = () => (
  <div style={{ ...fill, background: 'var(--osd-bg)', color: 'var(--osd-text)', padding: PX, position: 'relative' }}>
    <h2 style={{ fontFamily: 'var(--osd-font-display)', fontSize: 72, fontWeight: 900, margin: 0 }}>
      Models, Datasets &amp; Hallucination Rates
    </h2>
    <div style={{ display: 'flex', gap: 48, marginTop: 56 }}>
      <div style={{ flex: 1 }}>
        <div style={subHeading}>MODELS</div>
        <div style={{ fontSize: 32, fontWeight: 700, marginBottom: 8 }}>Llama-3.1-8B-Instruct</div>
        <p style={{ fontSize: 26, lineHeight: 1.5, color: muted }}>8B params · 32 layers · Meta</p>
        <div style={{ fontSize: 32, fontWeight: 700, marginTop: 28, marginBottom: 8 }}>Gemma-2-9B-IT</div>
        <p style={{ fontSize: 26, lineHeight: 1.5, color: muted }}>9B params · 42 layers · Google DeepMind</p>
      </div>
      <div style={{ flex: 1 }}>
        <div style={subHeading}>DATASETS</div>
        <div style={{ fontSize: 28, fontWeight: 700, marginBottom: 6 }}>BeliefBank Facts (BBF)</div>
        <p style={{ fontSize: 26, lineHeight: 1.5, color: muted, marginBottom: 16 }}>27,416 statements — Factual hallucinations - Extrinsic</p>
        <div style={{ fontSize: 28, fontWeight: 700, marginBottom: 6 }}>BeliefBank Constraints (BBC)</div>
        <p style={{ fontSize: 26, lineHeight: 1.5, color: muted, marginBottom: 16 }}>25,756 statements — Logical hallucinations - Instrinsic</p>
        <div style={{ fontSize: 28, fontWeight: 700, marginBottom: 6 }}>HaluEval (HE)</div>
        <p style={{ fontSize: 26, lineHeight: 1.5, color: muted }}>10,000 examples — Contextual hallucinations - Intrinsic</p>
      </div>
    </div>
    <div style={{ marginTop: 44 }}>
      <div style={{ display: 'flex', borderBottom: `2px solid var(--osd-accent)` }}>
        <div style={{ ...hrHeader, flex: 1, textAlign: 'left', paddingLeft: 0 }}>MODEL</div>
        <div style={{ ...hrHeader, flex: 1 }}>BBF</div>
        <div style={{ ...hrHeader, flex: 1 }}>BBC</div>
        <div style={{ ...hrHeader, flex: 1 }}>HE</div>
      </div>
      <div style={{ display: 'flex' }}>
        <div style={{ ...hrCell, flex: 1, textAlign: 'left', paddingLeft: 0, fontWeight: 600 }}>Llama-3.1-8B</div>
        <div style={{ ...hrCell, flex: 1, color: '#2f6f4e' }}>6.6%</div>
        <div style={{ ...hrCell, flex: 1, color: '#a13a3a' }}>56.0%</div>
        <div style={{ ...hrCell, flex: 1, color: '#b0812e' }}>23.9%</div>
      </div>
      <div style={{ display: 'flex' }}>
        <div style={{ ...hrCell, flex: 1, textAlign: 'left', paddingLeft: 0, fontWeight: 600 }}>Gemma-2-9B</div>
        <div style={{ ...hrCell, flex: 1, color: '#2f6f4e' }}>2.9%</div>
        <div style={{ ...hrCell, flex: 1, color: '#b0812e' }}>49.1%</div>
        <div style={{ ...hrCell, flex: 1, color: '#a13a3a' }}>27.5%</div>
      </div>
    </div>
    <Footer />
  </div>
);

// ============================================================
// PAGE — Dataset Example (BeliefBank Facts)
// ============================================================
const DSField = ({ label, children }: { label: string; children: React.ReactNode }) => (
  <div style={{ marginTop: 12 }}>
    <div style={{ fontSize: 15, fontWeight: 700, color: muted, letterSpacing: '0.1em', textTransform: 'uppercase' }}>{label}</div>
    <div style={{ fontSize: 25, marginTop: 3, color: 'var(--osd-text)', lineHeight: 1.25 }}>{children}</div>
  </div>
);

const ExampleCard = ({ tone, statement, answer, truth, verdict }: { tone: 'ok' | 'bad'; statement: string; answer: string; truth: string; verdict: string }) => {
  const color = tone === 'ok' ? '#2f6f4e' : '#a13a3a';
  return (
    <div style={{
      border: `2px solid ${color}`, borderRadius: 'var(--osd-radius)',
      padding: '18px 28px', background: tone === 'ok' ? 'rgba(47,111,78,0.05)' : 'rgba(161,58,58,0.05)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <div style={{ fontSize: 28, fontWeight: 800, color }}>{tone === 'ok' ? '✓' : '✗'}</div>
        <div style={{ fontSize: 24, fontWeight: 700, color, letterSpacing: '0.04em' }}>{verdict}</div>
      </div>
      <DSField label="Fact (prompt)">&laquo;{statement}&raquo;</DSField>
      <DSField label="Generated answer · Gemma-2-9B-IT"><strong>{answer}</strong></DSField>
      <DSField label="Ground truth"><strong>{truth}</strong></DSField>
    </div>
  );
};

// ============================================================
// PAGE 7 — Preliminary Study
// ============================================================
const PrelimStudy: Page = () => (
  <div style={{ ...fill, background: 'var(--osd-bg)', color: 'var(--osd-text)', padding: PX, position: 'relative' }}>
    <h2 style={{ fontFamily: 'var(--osd-font-display)', fontSize: '57px', fontWeight: 900, margin: 0 }}>
      Preliminary Study — Layer-Wise Component Accuracy
    </h2>
    <p style={{ fontSize: 26, color: muted, marginTop: 12 }}>
      Logistic Regression trained on individual layer components across all datasets
    </p>
    <div style={{ display: 'flex', justifyContent: 'center', marginTop: 36 }}>
      <div style={{ maxWidth: 1600, width: '100%' }}>
        <div style={{ fontSize: 28, fontWeight: 700, color: 'var(--osd-accent)', letterSpacing: '0.1em', marginBottom: 14, textAlign: 'center' }}>ATTENTION COMPONENT IN TRANSFORMER</div>
        <img src={miniAttn} style={{ width: '100%', height: 550, objectFit: 'contain', borderRadius: 'var(--osd-radius)', filter: chartFilter }} />
      </div>
    </div>
    <div style={{
      marginTop: 2, padding: '22px 32px',
      border: `1px solid ${borderColor}`, borderRadius: 'var(--osd-radius)',
      background: surface,
      fontSize: 26, lineHeight: 1.5, color: muted,
    }}>
      <strong style={{ color: 'var(--osd-text)' }}>Key insight:</strong>{' Intermediate layers are most informative — Llama peaks around layers 13–16, Gemma around 23–28'}
    </div>
    <Footer />
  </div>
);

const MethodChip = ({ highlight, children }: { highlight?: boolean; children: string }) => (
  <span
    style={{
      padding: '7px 18px', fontSize: 19, fontWeight: 600,
      border: `1px solid ${highlight ? 'var(--osd-accent)' : borderColor}`,
      borderRadius: 24,
      color: highlight ? 'var(--osd-accent)' : muted,
      background: highlight ? 'rgba(31,58,86,0.1)' : 'transparent',
    }}
  >
    {children}
  </span>
);

// ============================================================
// PAGE 8 — Detection Methodology (pipelines mirror thesis Fig. 3.8 & 3.22)
// ============================================================
const phaseTone = {
  blue:   { border: 'rgba(70,100,150,0.55)', bg: 'rgba(70,100,150,0.06)', text: '#3a5b8c' },
  violet: { border: 'rgba(130,90,160,0.55)', bg: 'rgba(130,90,160,0.06)', text: '#6b4a86' },
  green:  { border: 'rgba(70,140,90,0.55)',  bg: 'rgba(70,140,90,0.06)',  text: '#2f6f4e' },
  orange: { border: 'rgba(200,140,55,0.6)',  bg: 'rgba(200,140,55,0.07)', text: '#a9772a' },
  teal:   { border: 'rgba(55,150,150,0.55)', bg: 'rgba(55,150,150,0.06)', text: '#2c7a7a' },
} as const;

const PhasePanel = ({ tone, title, note, children }: { tone: keyof typeof phaseTone; title: string; note?: string; children: React.ReactNode }) => {
  const t = phaseTone[tone];
  return (
    <div style={{ border: `1.5px dashed ${t.border}`, background: t.bg, borderRadius: 10, padding: '10px 16px 12px', height: 200, boxSizing: 'border-box' }}>
      <div style={{ fontSize: 18, fontWeight: 800, color: t.text, marginBottom: 6 }}>{title}</div>
      <div style={{ display: 'flex', alignItems: 'stretch', gap: 4 }}>{children}</div>
      <div style={{ fontSize: 14, color: '#a13a3a', marginTop: 6, fontStyle: 'italic', minHeight: 15 }}>{note ? `↺ ${note}` : ''}</div>
    </div>
  );
};

const vecBar: React.CSSProperties = { width: 8, height: 30, borderRadius: 2, background: 'rgba(70,110,170,0.45)' };

const VecBox = ({ label }: { label: string }) => (
  <div style={{ flex: 0.85, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3 }}>
    <div style={{ height: 14 }} />
    <div style={{ minHeight: 66, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 6, lineHeight: '1.7', letterSpacing: '0.6px', fontWeight: '800', fontSize: '18px' }}>
      <div style={{ display: 'flex', gap: 3 }}>
        <span style={{ ...(vecBar), fontSize: '18px' }} /><span style={vecBar} /><span style={vecBar} /><span style={vecBar} />
      </div>
      <div style={{ fontSize: 13, color: muted, textAlign: 'center', lineHeight: 1.1 }}>{label}</div>
    </div>
  </div>
);

const TagBox = ({ tag, frozen, children }: { tag?: string; frozen?: boolean; children: React.ReactNode }) => (
  <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3 }}>
    <div style={{ fontSize: 13, fontWeight: 700, height: 14, color: frozen ? '#3a5b8c' : muted }}>{tag || ''}</div>
    <div style={{
      width: '100%', minHeight: 66, display: 'flex', alignItems: 'center', justifyContent: 'center', textAlign: 'center',
      border: `1.4px solid ${frozen ? 'rgba(58,91,140,0.55)' : borderColor}`, borderRadius: 8,
      background: frozen ? 'rgba(58,91,140,0.1)' : '#fff', padding: '12px 10px', fontSize: 18, fontWeight: 700, lineHeight: 1.15,
    }}>{children}</div>
  </div>
);

const ResultPill = ({ kind, children }: { kind: 'acc' | 'loss' | 'out'; children: React.ReactNode }) => {
  const c = kind === 'loss' ? { b: '#a13a3a', bg: 'rgba(161,58,58,0.12)' } : { b: '#2f6f4e', bg: 'rgba(47,111,78,0.14)' };
  return (
    <div style={{ flex: 0.65, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3 }}>
      <div style={{ height: 14 }} />
      <div style={{ minHeight: 66, display: 'flex', alignItems: 'center' }}>
        <div style={{ width: 60, height: 60, borderRadius: '50%', border: `2px solid ${c.b}`, background: c.bg, color: c.b, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 15, fontWeight: 800, textAlign: 'center', lineHeight: 1.05 }}>
          {children}
        </div>
      </div>
    </div>
  );
};

const FlowArrow = () => (
  <div style={{ flex: 0, display: 'flex', alignItems: 'center' }}>
    <div style={{ minHeight: 66, display: 'flex', alignItems: 'center', fontSize: 20, color: 'rgba(31,58,86,0.6)', fontWeight: 700, padding: '0 1px' }}>→</div>
  </div>
);

const colHeader: React.CSSProperties = { fontSize: 20, fontWeight: 800, letterSpacing: '0.04em', marginBottom: 12 };

const ProbingMethodology: Page = () => (
  <div style={{ ...fill, background: 'var(--osd-bg)', color: 'var(--osd-text)', padding: PX, position: 'relative' }}>
    <h2 style={{ fontFamily: 'var(--osd-font-display)', fontSize: 56, fontWeight: 900, margin: 0 }}>
      Detection Methodology
    </h2>
    <p style={{ fontSize: 24, color: muted, marginTop: 10, lineHeight: 1.4, maxWidth: 1560 }}>
      Eight approaches transfer a detector from a <span style={{ color: 'var(--osd-text)' }}>Trainer</span> to a
      <span style={{ color: 'var(--osd-text)' }}> Tester</span> model: seven baselines align the two activation spaces,
      while <span style={{ color: 'var(--osd-accent)' }}>One-For-All</span> shares a frozen classification head.
    </p>
    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 12 }}>
      <MethodChip>RidgeRegressor</MethodChip>
      <MethodChip>Procrustes</MethodChip>
      <MethodChip>CKA</MethodChip>
      <MethodChip>CCA</MethodChip>
      <MethodChip>AdapterMLP</MethodChip>
      <MethodChip>Full Non-linear</MethodChip>
      <MethodChip>Reduced Non-linear</MethodChip>
      <MethodChip highlight>One-For-All</MethodChip>
    </div>

    <div style={{ display: 'flex', gap: 40, marginTop: 22 }}>
      {/* Standard baseline — Fig. 3.8 (anonymized) */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 12 }}>
        <div style={{ ...colHeader, color: muted, fontSize: '19px', minHeight: 48 }}>{'STANDARD BASELINE (classifier and aligner change across different approaches)'}</div>
        <PhasePanel tone="blue" title="Phase 1 · Train classifier">
          <VecBox label="Trainer activations" />
          <FlowArrow />
          <TagBox>Classifier</TagBox>
          <FlowArrow />
          <ResultPill kind="acc">Output</ResultPill>
        </PhasePanel>
        <PhasePanel tone="violet" title="Phase 2 · Fit alignment (concordant data)">
          <VecBox label="Tester activations" />
          <FlowArrow />
          <TagBox>Aligner</TagBox>
          <FlowArrow />
          <VecBox label="Projected Tester activations on Trainer's space" />
        </PhasePanel>
        <PhasePanel tone="green" title="Phase 3 · Evaluate">
          <VecBox label="Tester test acts" />
          <FlowArrow />
          <TagBox>Aligner</TagBox>
          <FlowArrow />
          <TagBox>Classifier</TagBox>
          <FlowArrow />
          <ResultPill kind="acc">Output</ResultPill>
        </PhasePanel>
      </div>

      {/* One-For-All — Fig. 3.22 */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 12 }}>
        <div style={{ ...colHeader, color: 'var(--osd-accent)', fontSize: '19px', minHeight: 48 }}>ONE-FOR-ALL&nbsp;·&nbsp;proposed</div>
        <PhasePanel tone="blue" title="Phase 1 · Train classifier" note="backprop trains encoder + head">
          <VecBox label="Trainer activations" />
          <FlowArrow />
          <TagBox tag="Trainable">Encoder (Trainer)</TagBox>
          <FlowArrow />
          <TagBox tag="Trainable">Classification Head</TagBox>
          <FlowArrow />
          <ResultPill kind="out">Output</ResultPill>
        </PhasePanel>
        <PhasePanel tone="violet" title="Phase 2 · Adapt Tester" note="backprop trains encoder only">
          <VecBox label="Tester activations" />
          <FlowArrow />
          <TagBox tag="Trainable">Encoder (Tester)</TagBox>
          <FlowArrow />
          <TagBox tag="❄ Frozen" frozen>Head (from Trainer)</TagBox>
          <FlowArrow />
          <ResultPill kind="out">Output</ResultPill>
        </PhasePanel>
        <PhasePanel tone="green" title="Phase 3 · Evaluate">
          <VecBox label="Tester test acts" />
          <FlowArrow />
          <TagBox tag="❄ Frozen" frozen>Encoder (Tester)</TagBox>
          <FlowArrow />
          <TagBox tag="❄ Frozen" frozen>Head (from Trainer)</TagBox>
          <FlowArrow />
          <ResultPill kind="out">Output</ResultPill>
        </PhasePanel>
      </div>
    </div>
    <Footer />
  </div>
);

const methodBox: React.CSSProperties = {
  border: `1px solid ${borderColor}`,
  borderRadius: 'var(--osd-radius)',
  padding: '22px 26px',
  background: '#ffffff',
  height: 600,
  display: 'flex',
  flexDirection: 'column',
  justifyContent: 'center',
  gap: 16,
};

// ============================================================
// PAGE 9 — Probing: TruthX & SLiMTruth
// ============================================================
const ProbingTruthSLiM: Page = () => (
  <div style={{ ...fill, background: 'var(--osd-bg)', color: 'var(--osd-text)', padding: PX, position: 'relative' }}>
    <h2 style={{ fontFamily: 'var(--osd-font-display)', fontSize: 60, fontWeight: 900, margin: 0 }}>
      Mitigation Methods
    </h2>
    <p style={{ fontSize: 24, color: muted, marginTop: 10 }}>
      Procrustes was used as the aligner.
    </p>
    <div style={{ display: 'flex', gap: 48, marginTop: 32 }}>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 12 }}>
        <div style={{ fontSize: 32, fontWeight: 700, fontFamily: 'var(--osd-font-display)', marginBottom: 12 }}>
          TruthX
        </div>
        <div style={methodBox}>
          <img
            src={truthxDiagram}
            style={{ width: '100%', height: '100%', objectFit: 'contain' }}
          />
        </div>
      </div>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 12 }}>
        <div style={{ fontSize: 32, fontWeight: 700, fontFamily: 'var(--osd-font-display)', marginBottom: 12 }}>
          SLiMTruth
        </div>
        <div style={methodBox}>
          <img
            src={slimTruthDiagram}
            style={{ width: '100%', height: '100%', objectFit: 'contain' }}
          />
        </div>
      </div>
    </div>
    <Footer />
  </div>
);

// ============================================================
// PAGE 10 — Transfer Results: BBC & HE
// ============================================================
const TransferResults: Page = () => (
  <div style={{ ...fill, background: 'var(--osd-bg)', color: 'var(--osd-text)', padding: PX, position: 'relative' }}>
    <h2 style={{ fontFamily: 'var(--osd-font-display)', fontSize: 60, fontWeight: 900, margin: 0 }}>
      Transfer Detection Results
    </h2>
    <p style={{ fontSize: 22, color: muted, marginTop: 6, textAlign: 'center' }}>
      Gemma-2-9B (Trainer) → Llama-3.1-8B (Tester)
    </p>
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20, marginTop: 16 }}>
      <div>
        <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--osd-accent)', letterSpacing: '0.1em', marginBottom: 6, textAlign: 'center' }}>
          BELIEFBANK CONSTRAINTS (BBC)
        </div>
        <img src={bbcChart} style={{ width: '100%', height: 280, objectFit: 'contain', borderRadius: 'var(--osd-radius)', filter: chartFilter }} />
      </div>
      <div>
        <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--osd-accent)', letterSpacing: '0.1em', marginBottom: 6, textAlign: 'center' }}>
          HALUEVAL (HE)
        </div>
        <img src={heChart} style={{ width: '100%', height: 280, objectFit: 'contain', borderRadius: 'var(--osd-radius)', filter: chartFilter }} />
      </div>
    </div>
    <div style={{ marginTop: 24, padding: '18px 28px', border: `1px solid ${borderColor}`, borderRadius: 'var(--osd-radius)', background: surface, fontSize: '24px', lineHeight: '0.8', color: muted, textAlign: 'center', letterSpacing: '3.8px' }}>
      <strong style={{ color: 'var(--osd-text)' }}>BeliefBank Facts has similar performances to BeliefBank Constraint</strong>{'s'}<strong style={{ color: 'var(--osd-text)' }}>{''}</strong>{''}
    </div>
    <Footer />
  </div>
);

// ============================================================
// PAGE 11 — Transfer Results: Change Domain
// ============================================================
const TransferChangeDomain: Page = () => (
  <div style={{ ...fill, background: 'var(--osd-bg)', color: 'var(--osd-text)', padding: PX, position: 'relative' }}>
    <h2 style={{ fontFamily: 'var(--osd-font-display)', fontSize: 60, fontWeight: 900, margin: 0 }}>
      Transfer Detection Results
    </h2>
    <p style={{ fontSize: 26, color: muted, marginTop: 10, textAlign: 'center', letterSpacing: '0.08em' }}>
      Change domain
    </p>
    <div style={{ marginTop: 28, display: 'flex', justifyContent: 'center' }}>
      <img
        src={crossDomainGemmaToLlama}
        style={{ width: '100%', height: 620, objectFit: 'contain', borderRadius: 'var(--osd-radius)', filter: chartFilter }}
      />
    </div>
    <Footer />
  </div>
);

// ============================================================
// PAGE 12 — Mitigation Results
// ============================================================
const MitigationResults: Page = () => (
  <div style={{ ...fill, background: 'var(--osd-bg)', color: 'var(--osd-text)', padding: PX, position: 'relative' }}>
    <h2 style={{ fontFamily: 'var(--osd-font-display)', fontSize: 60, fontWeight: 900, margin: 0 }}>
      Mitigation Results — TruthX &amp; SLiMTruth
    </h2>
    <p style={{ fontSize: 22, color: muted, marginTop: 6, textAlign: 'center' }}>Gemma-2-9B-IT</p>
    <div style={{ display: 'flex', gap: 32, marginTop: 16 }}>
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: '18px', fontWeight: 700, color: 'var(--osd-accent)', letterSpacing: '1.8px', marginBottom: 8, textAlign: 'center', lineHeight: '1.5' }}>
          TRUTHX
        </div>
        <img src={gemmaResults} style={{ width: '100%', height: 520, objectFit: 'contain', borderRadius: 'var(--osd-radius)', filter: chartFilter }} />
      </div>
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--osd-accent)', letterSpacing: '0.1em', marginBottom: 8, textAlign: 'center' }}>
          SLIMTRUTH
        </div>
        <img src={gemmaSlimResults} style={{ width: '100%', height: 520, objectFit: 'contain', borderRadius: 'var(--osd-radius)', filter: chartFilter }} />
      </div>
    </div>
    <div style={{
      marginTop: 48, padding: '22px 32px',
      border: `1px solid ${borderColor}`, borderRadius: 'var(--osd-radius)',
      background: surface,
      fontSize: 26, lineHeight: 1.5, color: muted,
    }}>
      <strong style={{ color: 'var(--osd-text)' }}>SLiMTruth</strong>{' dominates in-domain (Llama on BBF: −89.9% HR). '}
      <strong style={{ color: 'var(--osd-text)' }}>TruthX</strong>{' transfers better cross-model (HaluEval: ~−37% HR in both directions). '}
    </div>
    <Footer />
  </div>
);

// ============================================================
// PAGE 13 — Findings & Future Directions
// ============================================================
const FindingsFuture: Page = () => (
  <div style={{ ...fill, background: 'var(--osd-bg)', color: 'var(--osd-text)', padding: PX, position: 'relative' }}>
    <div style={{ display: 'flex', gap: 56, height: '100%' }}>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'flex-start', paddingTop: 100 }}>
        <h2 style={{ fontFamily: 'var(--osd-font-display)', fontSize: 56, fontWeight: 900, margin: 0, color: 'var(--osd-accent)' }}>
          Key Findings
        </h2>
        <ul style={{ fontSize: 26, lineHeight: 1.6, color: muted, paddingLeft: 24, marginTop: 28 }}>
          <li style={{ marginBottom: 14 }}>
            <span style={{ color: 'var(--osd-text)' }}>One-For-All</span>{' is the strongest cross-model detector '}
          </li>
          <li style={{ marginBottom: 14 }}>
            <span style={{ color: 'var(--osd-text)' }}>Directional asymmetry</span>{' is pervasive: transfer quality depends heavily on trainer and target setting'}
          </li>
          <li style={{ marginBottom: 14 }}>
            <span style={{ color: 'var(--osd-text)' }}>SLiMTruth</span> dominates in-domain (up to −89.9% HR);{' '}
            <span style={{ color: 'var(--osd-text)' }}>TruthX</span> is more transferable cross-model (~−37% on HaluEval)
          </li>
          <li style={{ marginBottom: 14 }}>
            <span style={{ color: 'var(--osd-text)' }}>Contextual Hallucinations <span style={{ fontWeight: '300', color: '#4b535d' }}>tend to perform worse</span></span>{''}
          </li>
        </ul>
      </div>
      <div style={{ width: 1, background: borderColor, margin: '80px 0' }} />
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'flex-start', paddingTop: 100 }}>
        <h2 style={{ fontFamily: 'var(--osd-font-display)', fontSize: 56, fontWeight: 900, margin: 0, color: 'var(--osd-accent)' }}>
          Future Work
        </h2>
        <ul style={{ fontSize: 26, lineHeight: 1.6, color: muted, paddingLeft: 24, marginTop: 28 }}>
          <li style={{ marginBottom: 14 }}>
            <span style={{ color: 'var(--osd-text)' }}>Direction-aware transfer</span>: model source→target explicitly rather than
            assuming symmetric transfer
          </li>
          <li style={{ marginBottom: 14 }}>
            <span style={{ color: 'var(--osd-text)' }}>Adaptive steering strength</span>: scale intervention intensity to the estimated
            baseline error rate
          </li>
          <li style={{ marginBottom: 14 }}>
            <span style={{ color: 'var(--osd-text)' }}>Open Answer Format</span> Testing: extend beyond binary classification to more complex outputs, like free-form text generation
          </li>
        </ul>
      </div>
    </div>
    <Footer />
  </div>
);

// ============================================================
// PAGE 14 — Closing
// ============================================================
const Closing: Page = () => (
  <div style={{
    ...fill,
    background: 'var(--osd-bg)',
    color: 'var(--osd-text)',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
    padding: `0 ${PX}px`,
    position: 'relative',
  }}>
    <div style={{ width: 64, height: 2, background: 'var(--osd-accent)', marginBottom: 48 }} />
    <img src={catImg} style={{ height: 200, objectFit: 'contain', marginBottom: 24 }} />
    <h1 style={{ fontFamily: 'var(--osd-font-display)', fontSize: 120, fontWeight: 900, margin: 0, textAlign: 'center' }}>
      Thank You
    </h1>
    <p style={{ fontSize: 36, color: muted, marginTop: 32, textAlign: 'center' }}>
      Questions?
    </p>
    <div style={{
      marginTop: 64, display: 'flex', alignItems: 'center', gap: 20,
      fontSize: 24, color: muted,
    }}>
      <img src={unibaLogo} style={{ width: 64, opacity: 0.5, objectFit: 'cover', objectPosition: '50% 50%', objectViewBox: 'inset(90.46% 0% 0.39% 90.78%)' }} />
      <span>{''}</span>
    </div>
    <Footer />
  </div>
);

// ============================================================
// EXPORTS
// ============================================================
export const meta: SlideMeta = {
  title: 'Towards Cross-Model Transferability of Hallucination Detection and Mitigation in Large Language Models',
  createdAt: '2026-05-25T16:53:27.921Z',
};
export default [
  Cover,
  Background,
  Taxonomy,
  ProbingSteering,
  Objectives,
  ModelsDatasets,
  PrelimStudy,
  ProbingMethodology,
  ProbingTruthSLiM,
  TransferResults,
  TransferChangeDomain,
  MitigationResults,
  FindingsFuture,
  Closing,
] satisfies Page[];
