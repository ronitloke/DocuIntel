"""Shared visual system and static presentation components for Streamlit."""

from __future__ import annotations

import streamlit as st


VISUAL_SYSTEM_CSS = """
<style>
:root {
    --di-navy: #111827;
    --di-workspace: #f7f8fc;
    --di-card: #ffffff;
    --di-indigo: #6366f1;
    --di-indigo-dark: #4f46e5;
    --di-cyan: #06b6d4;
    --di-success: #10b981;
    --di-warning: #f59e0b;
    --di-danger: #ef4444;
    --di-text: #111827;
    --di-muted: #6b7280;
    --di-border: #e5e7eb;
}

[data-testid="stAppViewContainer"] {
    background: var(--di-workspace);
}

html,
body,
#root {
    min-height: 100%;
    height: auto;
}

body {
    min-height: 100vh;
    overflow-y: auto;
}

.stApp,
[data-testid="stAppViewContainer"] {
    position: relative;
    min-height: 100vh;
    height: auto;
    overflow: visible;
}

[data-testid="stMain"] {
    min-height: 100vh;
    height: auto;
    overflow: visible;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stMainBlockContainer"] {
    width: 100%;
    max-width: 1800px;
    margin-left: auto;
    margin-right: auto;
    box-sizing: border-box;
    min-width: 0;
    padding-top: 2.25rem;
    padding-left: clamp(20px, 1.5vw, 28px);
    padding-right: clamp(20px, 1.5vw, 28px);
    padding-bottom: 3rem;
}

[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    background: var(--di-navy);
}

section[data-testid="stSidebar"],
[data-testid="stSidebar"] {
    width: 280px !important;
    min-width: 280px !important;
    max-width: 280px !important;
    position: fixed !important;
    inset: 0 auto 0 0 !important;
    height: 100dvh !important;
    min-height: 100dvh !important;
    z-index: 1000 !important;
}

[data-testid="stAppViewContainer"] > section[data-testid="stSidebar"] + div {
    width: calc(100% - 280px) !important;
    max-width: none !important;
    margin-left: 280px !important;
    flex: 0 0 calc(100% - 280px) !important;
}

[data-testid="stSidebarContent"] {
    position: relative !important;
    height: 100dvh !important;
    min-height: 100dvh !important;
    box-sizing: border-box;
    overflow-y: auto;
}

[data-testid="stSidebarHeader"] {
    height: 24px !important;
    min-height: 24px !important;
}

[data-testid="stSidebar"] [class*="st-key-nav-"] {
    width: 100% !important;
    max-width: none !important;
}

[data-testid="stSidebar"] [data-testid="stButton"] {
    width: 100% !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #d1d5db;
}

[data-testid="stSidebar"] [data-testid="stButton"] > button:not([kind="primary"]) {
    background: transparent !important;
    border-color: transparent !important;
    color: #d1d5db !important;
}

[data-testid="stSidebar"] [data-testid="stButton"] > button:not([kind="primary"]):hover {
    border-color: rgba(255, 255, 255, 0.12) !important;
    background: rgba(255, 255, 255, 0.08) !important;
    color: #ffffff !important;
}

[data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"] {
    background: var(--di-indigo) !important;
    border-color: var(--di-indigo) !important;
    color: #ffffff !important;
}

[data-testid="stSidebar"] .stButton {
    margin: 0.15rem 0;
}

[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    min-height: 2.55rem;
    justify-content: flex-start;
    border: 1px solid transparent;
    border-radius: 0.7rem;
    background: transparent;
    color: #d1d5db;
    box-shadow: none;
    font-weight: 650;
    text-align: left;
}

[data-testid="stSidebar"] .stButton > button > div {
    width: 100% !important;
    justify-content: flex-start !important;
}

[data-testid="stSidebar"] .stButton > button > div > span {
    width: 100% !important;
    justify-content: flex-start !important;
    gap: 0.75rem !important;
}

[data-testid="stSidebar"] .stButton > button [data-testid="stIconMaterial"] {
    width: 1.35rem;
    color: currentColor;
    font-size: 1.2rem;
    text-align: center;
}

[data-testid="stSidebar"] .stButton > button [data-testid="stMarkdownContainer"] p {
    color: inherit !important;
    font-size: 0.91rem;
    font-weight: 650;
}

[data-testid="stSidebar"] .stButton > button:hover {
    border-color: rgba(255, 255, 255, 0.12);
    background: rgba(255, 255, 255, 0.08);
    color: #ffffff;
}

[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(100deg, #5b43e7 0%, #6d5df7 100%);
    color: #ffffff;
    box-shadow: 0 0.35rem 0.9rem rgba(99, 102, 241, 0.25);
}

[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: var(--di-indigo-dark);
}

[data-testid="stSidebar"] [data-testid="stAlert"] {
    min-height: 0;
    margin: 0.25rem 0;
    padding: 0.45rem 0.65rem;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 0.65rem;
    box-shadow: none;
}

[data-testid="stSidebar"] [data-testid="stAlert"] p {
    font-size: 0.8rem;
    line-height: 1.25;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--di-border);
    border-radius: 1rem;
    background: var(--di-card);
    box-shadow: 0 0.55rem 1.6rem rgba(17, 24, 39, 0.055);
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: #d9dcf8;
}

.stButton > button {
    border-radius: 0.65rem;
    font-weight: 600;
    transition: background-color 120ms ease, border-color 120ms ease, box-shadow 120ms ease;
}

.stButton > button[kind="primary"] {
    background: var(--di-indigo);
    border-color: var(--di-indigo);
    color: #ffffff;
    box-shadow: 0 0.35rem 0.8rem rgba(99, 102, 241, 0.18);
}

.stButton > button[kind="primary"]:hover {
    background: var(--di-indigo-dark);
    border-color: var(--di-indigo-dark);
}

.stButton > button[kind="secondary"] {
    background: #ffffff;
    border-color: #cfd4e4;
    color: var(--di-indigo-dark);
}

.stButton > button[kind="secondary"]:hover {
    border-color: var(--di-indigo);
    background: #f8f8ff;
}

[data-testid="stHorizontalBlock"] {
    min-width: 0;
}

[data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
[data-testid="stHorizontalBlock"] > [data-testid="stVerticalBlock"] {
    min-width: 0;
}

h1, h2, h3 {
    color: var(--di-text);
    letter-spacing: -0.025em;
}

h1 {
    font-size: clamp(2rem, 3.5vw, 3.2rem) !important;
    line-height: 1.08 !important;
}

h2 {
    font-size: clamp(1.35rem, 2vw, 1.8rem) !important;
}

h3 {
    font-size: clamp(1.08rem, 1.5vw, 1.3rem) !important;
}

.di-brand {
    padding: 0 0.2rem 1.9rem;
}

.di-brand-lockup {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    transform: translate(-0.15rem, -0.35rem);
}

.di-brand-mark {
    display: grid;
    width: 2.55rem;
    height: 2.55rem;
    flex: 0 0 2.55rem;
    place-items: center;
}

.di-brand-mark svg {
    width: 100%;
    height: 100%;
    overflow: visible;
}

.di-brand-name {
    color: #ffffff;
    font-size: 1.45rem;
    font-weight: 750;
    letter-spacing: -0.03em;
}

.di-brand-subtitle {
    margin-top: 0.25rem;
    color: #9ca3af;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}

.di-sidebar-section-label {
    margin: 0.5rem 0 0.45rem;
    color: #6b7280;
    font-size: 0.64rem;
    font-weight: 750;
    letter-spacing: 0.14em;
}

.di-welcome-row {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    margin: 0 0 0.45rem;
    color: var(--di-muted);
    font-size: 0.82rem;
    font-weight: 600;
}

.di-welcome-icon {
    display: inline-grid;
    width: 1.45rem;
    height: 1.45rem;
    place-items: center;
    border-radius: 0.5rem;
    background: #eef2ff;
    color: var(--di-indigo-dark);
    font-size: 0.82rem;
}

.di-eyebrow {
    margin: 0.5rem 0 0.85rem;
    color: var(--di-indigo-dark);
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 0.16em;
}

.di-hero-title {
    max-width: 42rem;
    margin: 0 0 1rem;
    color: var(--di-text);
    font-size: clamp(2rem, 2.55vw, 2.45rem) !important;
    font-weight: 780;
    letter-spacing: -0.055em;
    line-height: 1.08 !important;
}

.di-gradient-text {
    background: linear-gradient(105deg, var(--di-indigo-dark), var(--di-cyan));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

.di-hero-copy {
    max-width: 40rem;
    margin: 0 0 1.5rem;
    color: var(--di-muted);
    font-size: 1.05rem;
    line-height: 1.65;
}

.st-key-home-hero-grid {
    display: grid !important;
    grid-template-columns: minmax(0, 0.7fr) minmax(0, 1.3fr);
    align-items: center;
    gap: clamp(1.25rem, 2vw, 2rem);
    width: 100%;
    min-width: 0;
    min-height: 25.5rem;
    box-sizing: border-box;
    padding: 1.35rem 1.5rem 1.35rem;
    border: 1px solid #dfe3eb;
    border-radius: 1rem;
    background:
        radial-gradient(circle at 78% 38%, rgba(224, 231, 255, 0.42), transparent 34%),
        linear-gradient(135deg, rgba(255, 255, 255, 0.98), rgba(249, 250, 253, 0.96));
    box-shadow: 0 1rem 2.5rem rgba(17, 24, 39, 0.045);
}

.st-key-home-hero-grid > div {
    width: auto !important;
    min-width: 0;
}

.st-key-home-hero-copy,
.st-key-home-hero-pipeline {
    min-width: 0;
}

.di-pipeline-card {
    width: 100%;
    box-sizing: border-box;
    min-height: 0;
    padding: 0.1rem 0;
    border: 0;
    background: transparent;
    box-shadow: none;
}

.di-hero-actions {
    margin-bottom: 0.5rem;
}

.di-pipeline-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 0 0 0.85rem;
}

.di-pipeline-kicker {
    color: #4f46e5;
    font-size: 0.66rem;
    font-weight: 800;
    letter-spacing: 0.13em;
}

.di-pipeline-live {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    color: #059669;
    font-size: 0.68rem;
    font-weight: 700;
}

.di-pipeline-live::before {
    width: 0.4rem;
    height: 0.4rem;
    border-radius: 999px;
    background: var(--di-success);
    content: "";
}

.di-pipeline-diagram {
    --di-pipeline-columns: minmax(3.6rem, 1.18fr) auto minmax(3.25rem, 0.95fr) auto minmax(3.25rem, 0.95fr) auto minmax(5.6rem, 1.35fr) auto minmax(3.25rem, 0.95fr) auto minmax(3.25rem, 0.95fr) auto minmax(3.6rem, 1.18fr);
    padding: 0.15rem 0.2rem 0.05rem;
    background-image: radial-gradient(#dce3fa 0.75px, transparent 0.75px);
    background-position: center;
    background-size: 13px 13px;
}

.di-pipeline-rail {
    display: grid;
    grid-template-columns: var(--di-pipeline-columns);
    align-items: center;
    min-height: 2.75rem;
}

.di-pipeline-rail--top .di-pipeline-source,
.di-pipeline-rail--bottom .di-pipeline-source {
    grid-column: 7;
    justify-self: center;
}

.di-pipeline-rail--top .di-pipeline-rail-arrow,
.di-pipeline-rail--bottom .di-pipeline-rail-arrow {
    grid-column: 7;
    justify-self: center;
}

.di-pipeline-rail-arrow {
    color: var(--di-indigo);
    font-size: 1rem;
    font-weight: 800;
    line-height: 1;
}

.di-pipeline-source {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    width: 8.1rem;
    min-height: 2.75rem;
    padding: 0.35rem 0.55rem;
    box-sizing: border-box;
    border: 1px solid #d9ddea;
    border-radius: 0.65rem;
    background: rgba(255, 255, 255, 0.94);
    box-shadow: 0 0.35rem 0.8rem rgba(17, 24, 39, 0.04);
}

.di-pipeline-source strong {
    display: block;
    color: var(--di-text);
    font-size: 0.64rem;
    line-height: 1.2;
}

.di-pipeline-source small {
    display: block;
    margin-top: 0.12rem;
    color: var(--di-muted);
    font-size: 0.54rem;
    line-height: 1.2;
}

.di-pipeline-source--database .di-pipeline-source-icon {
    background: #eef2ff;
    color: #4f46e5;
}

.di-pipeline-source-icon {
    display: grid;
    width: 1.7rem;
    height: 1.7rem;
    flex: 0 0 1.7rem;
    place-items: center;
    border-radius: 0.5rem;
    background: #eef2ff;
    color: var(--di-indigo-dark);
    font-size: 0.95rem;
    font-weight: 800;
}

.di-pipeline-main-row {
    display: grid;
    grid-template-columns: var(--di-pipeline-columns);
    align-items: center;
    gap: 0.28rem;
    min-width: 0;
}

.di-pipeline-node {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-width: 0;
    min-height: 4.15rem;
    padding: 0.35rem 0.25rem;
    box-sizing: border-box;
    border: 1px solid #d9ddea;
    border-radius: 0.65rem;
    background: rgba(255, 255, 255, 0.92);
    text-align: center;
}

.di-pipeline-node strong {
    display: block;
    color: var(--di-text);
    font-size: 0.62rem;
    line-height: 1.2;
}

.di-pipeline-node small {
    display: block;
    margin-top: 0.16rem;
    color: var(--di-muted);
    font-size: 0.51rem;
    line-height: 1.2;
}

.di-pipeline-icon {
    display: grid;
    width: 1.65rem;
    height: 1.65rem;
    margin-bottom: 0.28rem;
    place-items: center;
    border-radius: 0.48rem;
    background: #eef2ff;
    color: var(--di-indigo-dark);
    font-size: 0.88rem;
    font-weight: 800;
}

.di-pipeline-connector {
    display: grid;
    place-items: center;
    color: #4f6fe8;
    font-size: 0.9rem;
    font-weight: 800;
}

.di-pipeline-core {
    position: relative;
    display: flex;
    min-width: 0;
    min-height: 7.1rem;
    align-items: center;
    justify-content: center;
    overflow: visible;
    border: 1px solid rgba(99, 102, 241, 0.72);
    border-radius: 1rem;
    background: radial-gradient(circle at 50% 45%, #312e81, #111827 73%);
    box-shadow: 0 0 0 0.25rem rgba(99, 102, 241, 0.08), 0 0.8rem 1.75rem rgba(79, 70, 229, 0.32), 0 0 2.3rem rgba(6, 182, 212, 0.3);
}

.di-pipeline-core::before,
.di-pipeline-core::after {
    position: absolute;
    width: 0.35rem;
    height: 0.35rem;
    border-radius: 999px;
    background: var(--di-cyan);
    box-shadow: 0 0 0.8rem rgba(6, 182, 212, 0.9);
    content: "";
}

.di-pipeline-core::before {
    top: 0.8rem;
    left: -0.55rem;
}

.di-pipeline-core::after {
    right: -0.55rem;
    bottom: 1rem;
}

.di-brain-svg {
    width: 4.25rem;
    height: 4.25rem;
    filter: drop-shadow(0 0 0.7rem rgba(129, 140, 248, 0.85));
}

.di-core-label {
    position: absolute;
    right: 0.25rem;
    bottom: 0.32rem;
    left: 0.25rem;
    color: #c7d2fe;
    font-size: 0.5rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-align: center;
    text-transform: uppercase;
}

.di-document-stack {
    position: relative;
    width: 5.4rem;
    height: 7.3rem;
    margin: 0 auto;
}

.di-document-sheet {
    position: absolute;
    top: 0.35rem;
    left: 0.4rem;
    width: 4rem;
    height: 6.1rem;
    box-sizing: border-box;
    border: 1px solid #d1d8e8;
    border-radius: 0.28rem;
    background: #ffffff;
    box-shadow: 0 0.35rem 0.65rem rgba(17, 24, 39, 0.08);
}

.di-document-sheet--back {
    top: 0.95rem;
    left: 0;
    opacity: 0.65;
    transform: rotate(-2deg);
}

.di-document-sheet--middle {
    top: 0.68rem;
    left: 0.2rem;
    opacity: 0.8;
    transform: rotate(1deg);
}

.di-document-sheet--front {
    display: block;
    overflow: hidden;
}

.di-document-sheet--front::after {
    position: absolute;
    top: 0;
    right: 0;
    width: 0;
    height: 0;
    border-top: 0.8rem solid #e4e8f2;
    border-left: 0.8rem solid transparent;
    content: "";
}

.di-pdf-badge {
    display: inline-block;
    margin: 0.28rem 0 0.28rem 0.35rem;
    padding: 0.12rem 0.2rem;
    border-radius: 0.15rem;
    background: #ef4444;
    color: white;
    font-size: 0.43rem;
    font-weight: 800;
}

.di-document-lines {
    display: block;
    width: 2.2rem;
    height: 0.25rem;
    margin: 0.18rem auto;
    border-radius: 0.15rem;
    background: #d7ddea;
    box-shadow: 0 0.42rem 0 #d7ddea, 0 0.84rem 0 #d7ddea;
}

.di-document-art {
    display: block;
    width: 2.25rem;
    height: 1.2rem;
    margin: 1.15rem auto 0;
    border-radius: 0.2rem;
    background: linear-gradient(145deg, #c7d2fe, #e0f2fe);
}

.di-pipeline-footer {
    margin-top: 0.55rem;
    color: #4b5563;
    font-size: 0.61rem;
    line-height: 1.35;
    text-align: center;
}

.di-section-heading {
    margin: 0;
}

.di-section-heading h2 {
    margin: 0 !important;
    font-size: 1.35rem !important;
    line-height: 1.2 !important;
}

.di-section-copy {
    margin-bottom: 1rem;
    color: var(--di-muted);
}

.di-kpi-label {
    margin-bottom: 0.4rem;
    color: var(--di-muted);
    font-size: 0.73rem;
    font-weight: 750;
    letter-spacing: 0.05em;
}

.di-kpi-icon {
    display: inline-grid;
    width: 4.1rem;
    height: 4.1rem;
    flex: 0 0 4.1rem;
    margin-bottom: 0;
    place-items: center;
    border-radius: 999px;
    font-size: 1.05rem;
    font-weight: 800;
}

.di-kpi-icon--violet { background: #f5f3ff; color: #7c3aed; }
.di-kpi-icon--blue { background: #eff6ff; color: #2563eb; }
.di-kpi-icon--cyan { background: #ecfeff; color: #0891b2; }
.di-kpi-icon--green { background: #ecfdf5; color: #059669; }

.di-kpi-icon svg,
.di-capability-icon svg {
    width: 1.65rem;
    height: 1.65rem;
    display: block;
}

.di-kpi-value {
    color: var(--di-text);
    font-size: 1.45rem;
    font-weight: 750;
}

.di-kpi-caption {
    margin-top: 0.2rem;
    color: var(--di-muted);
    font-size: 0.78rem;
}

.di-kpi-layout,
.di-capability-layout {
    display: flex;
    align-items: center;
    min-width: 0;
    height: 100%;
}

.di-kpi-layout {
    gap: 0.95rem;
    padding-left: 0.8rem;
    border-left: 2px solid #c4b5fd;
}

.di-kpi-layout--blue { border-left-color: #93c5fd; }
.di-kpi-layout--cyan { border-left-color: #67e8f9; }
.di-kpi-layout--green { border-left-color: #6ee7b7; }

.di-kpi-content,
.di-capability-content {
    min-width: 0;
}

.di-kpi-layout .di-kpi-label {
    margin-bottom: 0.3rem;
    color: var(--di-text);
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}

.di-kpi-layout .di-kpi-value {
    font-size: 1.65rem;
    line-height: 1.1;
}

.di-kpi-layout .di-kpi-caption {
    max-width: 9rem;
    line-height: 1.35;
}

.di-capability-layout {
    align-items: flex-start;
    gap: 0.9rem;
}

.di-capability-icon {
    display: inline-grid;
    width: 3.25rem;
    height: 3.25rem;
    flex: 0 0 3.25rem;
    place-items: center;
    border-radius: 999px;
    font-size: 1.15rem;
    font-weight: 800;
}

.di-capability-icon--violet { background: #f5f3ff; color: #7c3aed; }
.di-capability-icon--blue { background: #eff6ff; color: #2563eb; }
.di-capability-icon--cyan { background: #ecfeff; color: #0891b2; }
.di-capability-icon--green { background: #ecfdf5; color: #059669; }

.di-capability-icon svg {
    width: 1.75rem;
    height: 1.75rem;
}

.di-capability-title {
    color: var(--di-text);
    font-size: 0.98rem;
    font-weight: 700;
    line-height: 1.25;
}

.di-capability-description {
    margin-top: 0.45rem;
    color: var(--di-muted);
    font-size: 0.78rem;
    line-height: 1.45;
}

/* Documents page presentation. Keep this scope separate so the accepted Home layout stays stable. */
.st-key-documents-page {
    width: 100%;
    min-width: 0;
}

.st-key-documents-page .di-eyebrow {
    margin-top: 0;
}

.st-key-documents-page .di-page-title {
    margin: 0;
    color: var(--di-text);
    font-size: clamp(2rem, 3vw, 2.55rem) !important;
    font-weight: 780;
    letter-spacing: -0.055em;
    line-height: 1.08 !important;
}

.st-key-documents-page .di-page-subtitle {
    margin: 0.45rem 0 0;
    color: var(--di-muted);
    font-size: 1rem;
    line-height: 1.55;
}

.st-key-documents-page .di-card-heading {
    display: flex;
    align-items: flex-start;
    gap: 0.9rem;
    margin: 0 0 1.1rem;
}

.di-card-heading-icon {
    display: inline-grid;
    width: 3rem;
    height: 3rem;
    flex: 0 0 3rem;
    place-items: center;
    border-radius: 0.9rem;
    background: #eef2ff;
    color: var(--di-indigo);
}

.di-card-heading-icon svg {
    width: 1.65rem;
    height: 1.65rem;
}

.di-card-heading strong {
    display: block;
    color: var(--di-text);
    font-size: 1.15rem;
    font-weight: 750;
    line-height: 1.25;
}

.di-card-heading small {
    display: block;
    margin-top: 0.35rem;
    color: var(--di-muted);
    font-size: 0.88rem;
    line-height: 1.45;
}

.st-key-documents-page .st-key-documents-page-header {
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.15rem;
}

.st-key-documents-page .st-key-documents-page-header > [data-testid="stLayoutWrapper"]:first-child {
    min-width: 0;
}

.st-key-documents-page .st-key-documents-page-header [data-testid="stButton"] > button {
    min-height: 2.5rem;
    white-space: nowrap;
}

.st-key-documents-page .st-key-documents-upload-card,
.st-key-documents-page .st-key-documents-library-card,
.st-key-documents-page .st-key-documents-empty-state,
.st-key-documents-page .st-key-document-overview {
    min-width: 0;
}

.st-key-documents-page .st-key-documents-upload-card [data-testid="stVerticalBlockBorderWrapper"],
.st-key-documents-page .st-key-documents-library-card [data-testid="stVerticalBlockBorderWrapper"],
.st-key-documents-page .st-key-documents-empty-state [data-testid="stVerticalBlockBorderWrapper"],
.st-key-documents-page .st-key-document-overview [data-testid="stVerticalBlockBorderWrapper"] {
    padding: 1.35rem 1.45rem;
}

.st-key-documents-page .st-key-documents-upload-card [data-testid="stForm"] {
    padding: 1rem;
    border: 1px dashed #cfd4e4;
    border-radius: 0.9rem;
    background: linear-gradient(145deg, #fbfcff, #f7f8ff);
}

.di-upload-zone-copy {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    margin-bottom: 0.65rem;
}

.di-upload-zone-copy strong {
    color: var(--di-text);
    font-size: 0.95rem;
    font-weight: 750;
}

.di-upload-zone-copy span,
.di-upload-hints {
    color: var(--di-muted);
    font-size: 0.8rem;
    line-height: 1.45;
}

.di-upload-hints {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem 1.1rem;
    margin: 0.45rem 0 0.75rem;
}

.di-upload-hints span::before {
    margin-right: 0.35rem;
    color: var(--di-cyan);
    content: "•";
    font-weight: 800;
}

.st-key-documents-page .st-key-documents-upload-results [data-testid="stAlert"] {
    margin: 0.75rem 0 0;
    border-radius: 0.8rem;
    box-shadow: none;
}

.st-key-documents-section-heading,
.di-documents-section-heading {
    margin: 1.75rem 0 0.7rem;
}

.di-documents-section-heading h2 {
    margin: 0;
    color: var(--di-text);
    font-size: 1.35rem;
    font-weight: 760;
    letter-spacing: -0.035em;
}

.di-documents-section-heading p {
    margin: 0.3rem 0 0;
    color: var(--di-muted);
    font-size: 0.88rem;
}

.st-key-documents-library-card [data-testid="stSelectbox"] label {
    color: var(--di-text);
    font-size: 0.85rem;
    font-weight: 700;
}

.di-selected-document-heading h2 {
    margin: 0;
    overflow-wrap: anywhere;
    color: var(--di-text);
    font-size: clamp(1.3rem, 2vw, 1.75rem);
    font-weight: 760;
    letter-spacing: -0.035em;
    line-height: 1.2;
}

.di-selected-document-heading .di-eyebrow {
    margin-bottom: 0.45rem;
}

.st-key-document-overview-kpis {
    display: grid !important;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.85rem;
    width: 100%;
    margin-top: 1.15rem;
}

.st-key-document-overview-kpis > [data-testid="stLayoutWrapper"] {
    width: auto !important;
    min-width: 0;
}

.di-document-metric {
    display: flex;
    min-height: 4.4rem;
    flex-direction: column;
    justify-content: center;
    gap: 0.35rem;
    padding: 0.8rem 0.9rem;
    border-left: 2px solid #c4b5fd;
    border-radius: 0.1rem;
    background: #fafaff;
}

.st-key-document-overview-kpis > [data-testid="stLayoutWrapper"]:nth-child(2) .di-document-metric {
    border-left-color: #93c5fd;
}

.st-key-document-overview-kpis > [data-testid="stLayoutWrapper"]:nth-child(3) .di-document-metric {
    border-left-color: #67e8f9;
}

.st-key-document-overview-kpis > [data-testid="stLayoutWrapper"]:nth-child(4) .di-document-metric {
    border-left-color: #6ee7b7;
}

.di-document-metric-label {
    color: var(--di-muted);
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.di-document-metric-value {
    color: var(--di-text);
    font-size: 1.05rem;
    font-weight: 750;
    line-height: 1.2;
}

.di-status-badge {
    display: inline-flex;
    align-items: center;
    width: fit-content;
    padding: 0.18rem 0.55rem;
    border-radius: 999px;
    background: #ecfdf5;
    color: #047857;
    font-size: 0.78rem;
    font-weight: 750;
    text-transform: lowercase;
}

.st-key-document-actions {
    align-items: flex-end;
    gap: 0.75rem;
    margin: 1rem 0 1.25rem;
}

.st-key-document-actions [data-testid="stForm"] {
    display: flex;
    align-items: flex-end;
    gap: 0.75rem;
    min-width: 0;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
}

.st-key-document-actions [data-testid="stCheckbox"] {
    margin-bottom: 0.15rem;
}

.st-key-document-actions [data-testid="stFormSubmitButton"] button {
    border-color: #fecaca;
    color: #b91c1c;
}

.st-key-document-actions [data-testid="stFormSubmitButton"] button:hover {
    border-color: #ef4444;
    background: #fef2f2;
}

.di-page-view-heading {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 1rem;
    margin: 0.75rem 0;
}

.di-page-view-heading strong {
    color: var(--di-text);
    font-size: 1.05rem;
    font-weight: 750;
}

.di-page-view-heading span,
.di-page-extracted-label,
.di-chunk-meta {
    color: var(--di-muted);
    font-size: 0.8rem;
}

.di-page-extracted-label {
    margin: 0.9rem 0 0.25rem;
    font-weight: 700;
}

.di-chunk-meta {
    margin-bottom: 0.7rem;
    line-height: 1.5;
}

/* Analyze page presentation. Keep this scope separate from the accepted pages. */
.st-key-analyze-page {
    width: 100%;
    min-width: 0;
}

.st-key-analyze-page .di-eyebrow {
    margin-top: 0;
}

.st-key-analyze-page .di-page-title {
    margin: 0;
    color: var(--di-text);
    font-size: clamp(2rem, 3vw, 2.55rem) !important;
    font-weight: 780;
    letter-spacing: -0.055em;
    line-height: 1.08 !important;
}

.st-key-analyze-page .di-page-subtitle {
    max-width: 52rem;
    margin: 0.45rem 0 0;
    color: var(--di-muted);
    font-size: 1rem;
    line-height: 1.55;
}

.st-key-analyze-page .st-key-analyze-page-header {
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.25rem;
}

.st-key-analyze-page .st-key-analyze-page-header-copy {
    min-width: 0;
}

.di-analyze-header-icon {
    display: inline-grid;
    width: 4.15rem;
    height: 4.15rem;
    flex: 0 0 4.15rem;
    place-items: center;
    border: 1px solid #c7d2fe;
    border-radius: 1.2rem;
    background: linear-gradient(145deg, #eef2ff, #ecfeff);
    color: var(--di-indigo-dark);
    box-shadow: 0 0.65rem 1.6rem rgba(79, 70, 229, 0.1);
}

.di-analyze-header-icon svg {
    width: 2rem;
    height: 2rem;
}

.st-key-analyze-page [data-testid="stTabs"] [role="tablist"] {
    gap: 1.45rem;
    border-bottom: 1px solid var(--di-border);
}

.st-key-analyze-page [data-testid="stTabs"] button[role="tab"] {
    min-height: 2.85rem;
    padding: 0.25rem 0.05rem 0.7rem;
    border-bottom: 2px solid transparent;
    color: #4b5563;
    font-size: 0.88rem;
    font-weight: 650;
}

.st-key-analyze-page [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    border-bottom-color: var(--di-indigo);
    color: var(--di-indigo-dark);
    font-weight: 760;
}

.st-key-analyze-page .st-key-analyze-document-context,
.st-key-analyze-page .st-key-analyze-summary-card,
.st-key-analyze-page .st-key-analyze-classification-card,
.st-key-analyze-page .st-key-analyze-extraction-card,
.st-key-analyze-page .st-key-analyze-tables-card,
.st-key-analyze-page .st-key-analyze-empty-state {
    box-sizing: border-box;
    min-width: 0;
    padding: 1.25rem 1.35rem;
    border-color: #e0e4ef;
    background: #ffffff;
    box-shadow: 0 0.55rem 1.4rem rgba(17, 24, 39, 0.035);
}

.st-key-analyze-page .st-key-analyze-document-context {
    margin-bottom: 1.2rem;
}

.st-key-analyze-page .st-key-analyze-document-context [data-testid="stSelectbox"] {
    margin-bottom: 0.9rem;
}

.st-key-analyze-page .st-key-analyze-document-context [data-testid="stSelectbox"] label {
    color: var(--di-text);
    font-size: 0.85rem;
    font-weight: 700;
}

.di-selected-document-heading h2 {
    margin: 0;
    overflow-wrap: anywhere;
    color: var(--di-text);
    font-size: clamp(1.3rem, 2vw, 1.75rem);
    font-weight: 760;
    letter-spacing: -0.035em;
    line-height: 1.2;
}

.di-selected-document-heading .di-eyebrow {
    margin-bottom: 0.45rem;
}

.st-key-analyze-document-kpis {
    display: grid !important;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.85rem;
    width: 100%;
    margin-top: 1.1rem;
}

.st-key-analyze-document-kpis > [data-testid="stLayoutWrapper"] {
    width: auto !important;
    min-width: 0;
}

.di-analyze-metric {
    display: flex;
    min-height: 4.3rem;
    flex-direction: column;
    justify-content: center;
    gap: 0.35rem;
    padding: 0.8rem 0.9rem;
    border-left: 2px solid #c4b5fd;
    background: #fafaff;
}

.di-analyze-metric--blue { border-left-color: #93c5fd; }
.di-analyze-metric--cyan { border-left-color: #67e8f9; }
.di-analyze-metric--green { border-left-color: #6ee7b7; }

.di-analyze-metric span {
    color: var(--di-muted);
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.di-analyze-metric strong {
    overflow-wrap: anywhere;
    color: var(--di-text);
    font-size: 1.05rem;
    font-weight: 750;
    line-height: 1.2;
}

.di-analyze-card-heading {
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
    margin-bottom: 1rem;
}

.di-analyze-card-icon {
    display: inline-grid;
    width: 2.45rem;
    height: 2.45rem;
    flex: 0 0 2.45rem;
    place-items: center;
    border-radius: 0.75rem;
    background: #eef2ff;
    color: var(--di-indigo-dark);
}

.di-analyze-card-icon--blue { background: #eff6ff; color: #2563eb; }
.di-analyze-card-icon--cyan { background: #ecfeff; color: #0891b2; }
.di-analyze-card-icon--green { background: #ecfdf5; color: #059669; }

.di-analyze-card-icon svg {
    width: 1.3rem;
    height: 1.3rem;
}

.di-analyze-card-heading h2 {
    margin: 0;
    color: var(--di-text);
    font-size: 1.25rem;
    font-weight: 760;
    letter-spacing: -0.035em;
    line-height: 1.25;
}

.di-analyze-card-heading p {
    margin: 0.3rem 0 0;
    color: var(--di-muted);
    font-size: 0.86rem;
    line-height: 1.45;
}

.st-key-analyze-summary-card [data-testid="stForm"],
.st-key-analyze-classification-card [data-testid="stForm"],
.st-key-analyze-extraction-card [data-testid="stForm"],
.st-key-analyze-tables-card [data-testid="stForm"] {
    min-width: 0;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
}

.st-key-analyze-summary-card [data-testid="stFormSubmitButton"] button,
.st-key-analyze-classification-card [data-testid="stFormSubmitButton"] button,
.st-key-analyze-extraction-card [data-testid="stFormSubmitButton"] button,
.st-key-analyze-tables-card [data-testid="stFormSubmitButton"] button {
    min-height: 2.55rem;
}

.st-key-analyze-summary-result,
.st-key-analyze-classification-result,
.st-key-analyze-table-result,
[class*="st-key-analyze-field-result-"] {
    box-sizing: border-box;
    min-width: 0;
    margin-top: 1rem;
    padding: 1rem 1.05rem;
    border-color: #dbe3ff;
    background: #fbfcff;
    box-shadow: none;
}

.st-key-analyze-summary-result {
    border-color: #c7d2fe;
    background: linear-gradient(145deg, #ffffff, #f7f8ff 75%, #effcff);
}

.di-result-kicker,
.di-field-value-label,
.di-table-preview-label,
.di-table-result-label {
    color: var(--di-indigo-dark);
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.di-result-kicker {
    margin-bottom: 0.55rem;
}

.st-key-analyze-classification-result {
    border-color: #bfdbfe;
    background: #f8fbff;
}

.di-classification-label {
    display: inline-flex;
    max-width: 100%;
    margin: 0.15rem 0 0.75rem;
    padding: 0.45rem 0.7rem;
    overflow-wrap: anywhere;
    border: 1px solid #bfdbfe;
    border-radius: 0.7rem;
    background: #eff6ff;
    color: #1d4ed8;
    font-size: 1rem;
    font-weight: 750;
}

.di-analysis-evidence-heading {
    display: flex;
    align-items: flex-start;
    gap: 0.7rem;
    margin: 1.25rem 0 0.7rem;
}

.di-analysis-evidence-heading .di-analyze-card-icon {
    width: 2.1rem;
    height: 2.1rem;
    flex-basis: 2.1rem;
}

.di-analysis-evidence-heading h3 {
    margin: 0;
    color: var(--di-text);
    font-size: 1rem;
    font-weight: 760;
}

.di-analysis-evidence-heading p {
    margin: 0.25rem 0 0;
    color: var(--di-muted);
    font-size: 0.78rem;
}

.di-analysis-source-heading {
    display: flex;
    align-items: flex-start;
    gap: 0.7rem;
    margin-bottom: 0.7rem;
}

.di-analysis-source-heading strong,
.di-analysis-source-heading span {
    display: block;
    overflow-wrap: anywhere;
}

.di-analysis-source-heading strong {
    color: var(--di-text);
    font-size: 0.88rem;
}

.di-analysis-source-heading span {
    margin-top: 0.2rem;
    color: var(--di-muted);
    font-size: 0.76rem;
}

.di-extraction-field-heading {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 0.8rem;
    margin-bottom: 0.75rem;
}

.di-extraction-field-heading strong {
    display: block;
    overflow-wrap: anywhere;
    color: var(--di-text);
    font-size: 0.95rem;
    font-weight: 750;
}

.di-field-status {
    display: inline-flex;
    flex: 0 0 auto;
    padding: 0.22rem 0.55rem;
    border-radius: 999px;
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.05em;
}

.di-field-status--found {
    background: #ecfdf5;
    color: #047857;
}

.di-field-status--not-found {
    background: #f3f4f6;
    color: #6b7280;
}

.di-field-value-label,
.di-table-result-label {
    margin-bottom: 0.3rem;
    color: var(--di-muted);
    letter-spacing: 0.08em;
}

.st-key-analyze-tables-card [data-testid="stTable"] {
    max-width: 100%;
    overflow-x: auto;
}

.di-table-preview-label {
    margin: 1rem 0 0.45rem;
}

.di-table-examples {
    margin: 0.1rem 0 0.75rem;
    color: var(--di-muted);
    font-size: 0.78rem;
    line-height: 1.55;
}

.di-table-examples strong {
    color: var(--di-text);
    font-weight: 750;
}

.st-key-analyze-table-empty-state {
    box-sizing: border-box;
    margin-top: 1rem;
    padding: 1rem 1.05rem;
    border-color: #e0e4ef;
    background: #fbfcff;
    box-shadow: none;
}

.di-table-empty-heading {
    margin-bottom: 0.45rem;
    color: var(--di-text);
    font-size: 0.95rem;
    font-weight: 750;
}

.st-key-analyze-page [data-testid="stExpander"] {
    margin: 0.7rem 0;
    border-color: #e0e4ef;
    border-radius: 0.85rem;
    background: rgba(255, 255, 255, 0.72);
}

.st-key-analyze-page [data-testid="stExpander"] summary {
    color: var(--di-text);
    font-size: 0.86rem;
    font-weight: 700;
}

/* Compare page presentation. Keep comparison styling isolated from accepted pages. */
.st-key-compare-page {
    width: 100%;
    min-width: 0;
}

.st-key-compare-page .di-eyebrow {
    margin-top: 0;
}

.st-key-compare-page .di-page-title {
    margin: 0;
    color: var(--di-text);
    font-size: clamp(2rem, 3vw, 2.55rem) !important;
    font-weight: 780;
    letter-spacing: -0.055em;
    line-height: 1.08 !important;
}

.st-key-compare-page .di-page-subtitle {
    max-width: 54rem;
    margin: 0.45rem 0 0;
    color: var(--di-muted);
    font-size: 1rem;
    line-height: 1.55;
}

.st-key-compare-page .st-key-compare-page-header {
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.25rem;
}

.st-key-compare-page .st-key-compare-page-header-copy {
    min-width: 0;
}

.di-compare-header-icon {
    display: inline-grid;
    width: 4.15rem;
    height: 4.15rem;
    flex: 0 0 4.15rem;
    place-items: center;
    border: 1px solid #c7d2fe;
    border-radius: 1.2rem;
    background: linear-gradient(145deg, #eef2ff, #ecfeff);
    color: var(--di-indigo-dark);
    box-shadow: 0 0.65rem 1.6rem rgba(79, 70, 229, 0.1);
}

.di-compare-header-icon svg {
    width: 2rem;
    height: 2rem;
}

.st-key-compare-page .st-key-compare-setup-card,
.st-key-compare-page .st-key-compare-result,
.st-key-compare-page .st-key-compare-summary-card,
.st-key-compare-page [class*="st-key-compare-section-"],
.st-key-compare-page .st-key-compare-empty-state,
.st-key-compare-page .st-key-compare-table-empty-state {
    box-sizing: border-box;
    min-width: 0;
    padding: 1.3rem 1.4rem;
    border-color: #e0e4ef;
    background: #ffffff;
    box-shadow: 0 0.55rem 1.4rem rgba(17, 24, 39, 0.035);
}

.st-key-compare-page .st-key-compare-setup-card {
    margin-bottom: 1.25rem;
}

.st-key-compare-page .st-key-compare-setup-card [data-testid="stForm"] {
    min-width: 0;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
}

.di-compare-card-heading,
.di-compare-section-heading {
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
}

.di-compare-card-heading {
    margin-bottom: 1.15rem;
}

.di-compare-card-icon,
.di-compare-section-icon {
    display: inline-grid;
    flex: 0 0 auto;
    place-items: center;
    border-radius: 0.8rem;
    background: #eef2ff;
    color: var(--di-indigo-dark);
}

.di-compare-card-icon {
    width: 2.5rem;
    height: 2.5rem;
}

.di-compare-card-icon svg {
    width: 1.35rem;
    height: 1.35rem;
}

.di-compare-card-heading h2,
.di-compare-section-heading h2 {
    margin: 0;
    color: var(--di-text);
    font-size: 1.25rem;
    font-weight: 760;
    letter-spacing: -0.035em;
    line-height: 1.25;
}

.di-compare-card-heading p,
.di-compare-section-heading p {
    margin: 0.28rem 0 0;
    color: var(--di-muted);
    font-size: 0.85rem;
    line-height: 1.45;
}

.di-compare-side-label {
    display: flex;
    min-height: 3.1rem;
    flex-direction: column;
    justify-content: center;
    gap: 0.25rem;
    margin-bottom: 0.35rem;
    padding-left: 0.8rem;
    border-left: 3px solid #818cf8;
}

.di-compare-side-label--target {
    border-left-color: #22d3ee;
}

.di-compare-side-label span,
.di-compare-section-kicker,
.di-compare-result-kicker,
.di-compare-evidence-label,
.di-table-change-heading span {
    color: var(--di-indigo-dark);
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.di-compare-side-label--target span {
    color: #0891b2;
}

.di-compare-side-label small {
    color: var(--di-muted);
    font-size: 0.76rem;
}

.st-key-compare-setup-card [data-testid="stHorizontalBlock"] {
    min-width: 0;
}

.st-key-compare-setup-card [data-testid="stHorizontalBlock"] > [data-testid="column"] {
    min-width: 0;
}

.di-compare-direction-note {
    margin: 0.9rem 0 1.05rem;
    padding: 0.75rem 0.9rem;
    border: 1px solid #e0e7ff;
    border-radius: 0.7rem;
    background: #f8f9ff;
    color: var(--di-muted);
    font-size: 0.78rem;
    line-height: 1.5;
}

.di-compare-direction-note span {
    color: var(--di-text);
    font-weight: 750;
}

.st-key-compare-options {
    display: grid !important;
    grid-template-columns: minmax(11rem, 1.05fr) repeat(3, minmax(10rem, 1fr));
    align-items: end;
    gap: 0.85rem;
    width: 100%;
    min-width: 0;
}

.st-key-compare-options > [data-testid="stLayoutWrapper"] {
    width: auto !important;
    min-width: 0;
}

.st-key-compare-options [data-testid="stCheckbox"] {
    min-height: 2.75rem;
    align-items: center;
}

.st-key-compare-options [data-testid="stCheckbox"] label {
    color: var(--di-text);
    font-size: 0.78rem;
    font-weight: 650;
    line-height: 1.35;
}

.st-key-compare-setup-card [data-testid="stFormSubmitButton"] {
    margin-top: 1rem;
}

.st-key-compare-setup-card [data-testid="stFormSubmitButton"] button {
    min-height: 2.65rem;
    padding: 0.55rem 1rem;
    border-color: var(--di-indigo-dark);
    background: var(--di-indigo-dark);
    color: #ffffff;
    font-weight: 750;
}

.st-key-compare-setup-card [data-testid="stFormSubmitButton"] button:hover {
    border-color: #4338ca;
    background: #4338ca;
}

.di-compare-result-kicker {
    margin-bottom: 0.7rem;
}

.st-key-compare-result {
    margin-bottom: 1.2rem;
    border-color: #c7d2fe;
    background: linear-gradient(145deg, #ffffff, #fafaff 75%, #effcff);
}

.st-key-compare-result-pair {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
    align-items: center;
    gap: 0.8rem;
    width: 100%;
    min-width: 0;
    margin-bottom: 1.15rem;
}

.st-key-compare-result-pair > [data-testid="stLayoutWrapper"] {
    width: auto !important;
    min-width: 0;
}

.di-compare-document-chip {
    display: flex;
    min-width: 0;
    flex-direction: column;
    gap: 0.3rem;
    padding: 0.75rem 0.9rem;
    border: 1px solid #dbe3ff;
    border-radius: 0.75rem;
    background: rgba(255, 255, 255, 0.82);
}

.di-compare-document-chip--target {
    border-color: #bae6fd;
    background: #f5fdff;
}

.di-compare-document-chip span {
    color: var(--di-muted);
    font-size: 0.66rem;
    font-weight: 800;
    letter-spacing: 0.1em;
}

.di-compare-document-chip--target span {
    color: #0891b2;
}

.di-compare-document-chip strong {
    overflow-wrap: anywhere;
    color: var(--di-text);
    font-size: 0.92rem;
    line-height: 1.35;
}

.di-compare-pair-divider {
    display: inline-grid;
    width: 2.15rem;
    height: 2.15rem;
    place-items: center;
    border: 1px solid #dbe3ff;
    border-radius: 50%;
    background: #ffffff;
    color: var(--di-indigo-dark);
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.st-key-compare-stats-grid {
    display: grid !important;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 0.75rem;
    width: 100%;
    min-width: 0;
}

.st-key-compare-stats-grid > [data-testid="stLayoutWrapper"] {
    width: auto !important;
    min-width: 0;
}

.st-key-compare-stats-grid [data-testid="stVerticalBlockBorderWrapper"] {
    height: 100%;
    box-sizing: border-box;
    border-color: #e5e7eb;
    box-shadow: none;
}

.di-compare-stat {
    display: flex;
    min-height: 5rem;
    align-items: center;
    gap: 0.65rem;
    padding: 0.75rem 0.8rem;
}

.di-compare-stat-symbol {
    display: inline-grid;
    width: 2rem;
    height: 2rem;
    flex: 0 0 2rem;
    place-items: center;
    border-radius: 0.65rem;
    background: #f3f4f6;
    color: #6b7280;
    font-size: 1.1rem;
    font-weight: 800;
}

.di-compare-stat strong,
.di-compare-stat small {
    display: block;
}

.di-compare-stat strong {
    color: var(--di-text);
    font-size: 1.35rem;
    font-weight: 780;
    line-height: 1.1;
}

.di-compare-stat small {
    margin-top: 0.25rem;
    color: var(--di-muted);
    font-size: 0.72rem;
    font-weight: 700;
    line-height: 1.2;
}

.di-compare-stat--green .di-compare-stat-symbol { background: #ecfdf5; color: #059669; }
.di-compare-stat--red .di-compare-stat-symbol { background: #fff1f2; color: #e11d48; }
.di-compare-stat--indigo .di-compare-stat-symbol { background: #eef2ff; color: var(--di-indigo-dark); }
.di-compare-stat--cyan .di-compare-stat-symbol { background: #ecfeff; color: #0891b2; }

.st-key-compare-summary-card {
    margin-bottom: 1.2rem;
    border-color: #c7d2fe;
}

.di-compare-summary-card,
.di-compare-section-heading {
    min-width: 0;
}

.di-compare-section-heading {
    margin-bottom: 1rem;
}

.di-compare-section-heading--green .di-compare-section-icon { background: #ecfdf5; color: #059669; }
.di-compare-section-heading--red .di-compare-section-icon { background: #fff1f2; color: #e11d48; }
.di-compare-section-heading--neutral .di-compare-section-icon { background: #f3f4f6; color: #6b7280; }
.di-compare-section-heading--indigo .di-compare-section-icon { background: #eef2ff; color: var(--di-indigo-dark); }

.di-compare-section-icon {
    width: 2.3rem;
    height: 2.3rem;
    font-size: 1.05rem;
    font-weight: 800;
}

.di-compare-section-icon svg {
    width: 1.25rem;
    height: 1.25rem;
}

.di-compare-section-heading h2 em {
    display: inline-grid;
    min-width: 1.45rem;
    height: 1.45rem;
    margin-left: 0.25rem;
    place-items: center;
    border-radius: 999px;
    background: #f3f4f6;
    color: var(--di-muted);
    font-size: 0.72rem;
    font-style: normal;
    letter-spacing: 0;
    vertical-align: 0.12rem;
}

.di-compare-change-meta {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.8rem;
    margin-bottom: 0.75rem;
}

.di-compare-change-meta span {
    color: var(--di-text);
    font-size: 0.8rem;
    font-weight: 800;
}

.di-compare-change-meta small {
    color: var(--di-muted);
    font-size: 0.74rem;
}

[class*="st-key-compare-change-body-"] {
    min-width: 0;
}

[class*="st-key-compare-change-body-"] [data-testid="stHorizontalBlock"] {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.85rem;
    min-width: 0;
}

[class*="st-key-compare-change-body-"] [data-testid="stHorizontalBlock"] > [data-testid="column"] {
    width: auto !important;
    min-width: 0;
    padding: 0.9rem;
    border: 1px solid #e5e7eb;
    border-radius: 0.75rem;
    background: #fbfcff;
}

.di-compare-side-heading,
.di-compare-evidence-label {
    margin-bottom: 0.45rem;
}

.di-compare-side-heading {
    color: var(--di-indigo-dark);
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.1em;
}

.di-compare-side-heading--target {
    color: #0891b2;
}

.di-compare-content-text {
    min-height: 3.2rem;
    margin-bottom: 0.9rem;
    overflow-wrap: anywhere;
    color: var(--di-text);
    font-size: 0.88rem;
    line-height: 1.55;
    white-space: normal;
}

.di-compare-content-text--empty {
    color: #9ca3af;
    font-style: italic;
}

.di-compare-evidence-label {
    color: var(--di-muted);
    font-size: 0.64rem;
    letter-spacing: 0.1em;
}

.di-compare-source {
    display: flex;
    min-width: 0;
    align-items: flex-start;
    gap: 0.55rem;
    margin: 0.45rem 0;
}

.di-compare-source-badge {
    display: inline-grid;
    min-width: 1.9rem;
    min-height: 1.45rem;
    flex: 0 0 auto;
    place-items: center;
    padding: 0.15rem 0.35rem;
    border-radius: 0.45rem;
    background: #eef2ff;
    color: var(--di-indigo-dark);
    font-size: 0.66rem;
    font-weight: 800;
}

.di-compare-source strong,
.di-compare-source small {
    display: block;
    overflow-wrap: anywhere;
}

.di-compare-source strong {
    color: var(--di-text);
    font-size: 0.75rem;
}

.di-compare-source small {
    margin-top: 0.15rem;
    color: var(--di-muted);
    font-size: 0.68rem;
    line-height: 1.4;
}

[class*="st-key-compare-table-detail-"] {
    box-sizing: border-box;
    min-width: 0;
    margin-top: 0.9rem;
    padding: 0.85rem 0.9rem;
    border-color: #bae6fd;
    background: #f7fdff;
    box-shadow: none;
}

.di-table-change-heading {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 0.8rem;
    margin-bottom: 0.65rem;
}

.di-table-change-heading span {
    color: #0891b2;
}

.di-table-change-heading strong {
    color: var(--di-text);
    font-size: 0.82rem;
}

[class*="st-key-compare-table-detail-grid-"] {
    display: grid !important;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.55rem;
    width: 100%;
    min-width: 0;
}

[class*="st-key-compare-table-detail-grid-"] > [data-testid="stLayoutWrapper"] {
    width: auto !important;
    min-width: 0;
}

.di-table-change-cell {
    min-width: 0;
    padding: 0.55rem 0.6rem;
    border-radius: 0.55rem;
    background: rgba(255, 255, 255, 0.78);
}

.di-table-change-cell small,
.di-table-change-cell strong {
    display: block;
    overflow-wrap: anywhere;
}

.di-table-change-cell small {
    margin-bottom: 0.25rem;
    color: var(--di-muted);
    font-size: 0.64rem;
    font-weight: 750;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.di-table-change-cell strong {
    color: var(--di-text);
    font-size: 0.78rem;
    line-height: 1.35;
}

.di-table-change-row-label {
    margin: 0.75rem 0 0.35rem;
    color: var(--di-text);
    font-size: 0.72rem;
    font-weight: 750;
}

.st-key-compare-page [data-testid="stExpander"] {
    margin: 0.65rem 0;
    border-color: #e0e4ef;
    border-radius: 0.8rem;
    background: rgba(255, 255, 255, 0.76);
}

.st-key-compare-page [data-testid="stExpander"] summary {
    color: var(--di-text);
    font-size: 0.84rem;
    font-weight: 700;
}

.di-compare-empty-heading,
.di-compare-table-empty-heading {
    margin-bottom: 0.45rem;
    color: var(--di-text);
    font-size: 0.96rem;
    font-weight: 750;
}

@media (max-width: 900px) {
    .st-key-compare-options {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .st-key-compare-stats-grid {
        grid-template-columns: repeat(3, minmax(0, 1fr));
    }
}

@media (max-width: 560px) {
    .st-key-compare-page .st-key-compare-page-header {
        align-items: flex-start;
        flex-direction: column;
    }

    .st-key-compare-page .st-key-compare-setup-card,
    .st-key-compare-page .st-key-compare-result,
    .st-key-compare-page .st-key-compare-summary-card,
    .st-key-compare-page [class*="st-key-compare-section-"],
    .st-key-compare-page .st-key-compare-empty-state,
    .st-key-compare-page .st-key-compare-table-empty-state {
        padding: 1rem;
    }

    .st-key-compare-options,
    .st-key-compare-stats-grid,
    [class*="st-key-compare-table-detail-grid-"] {
        grid-template-columns: minmax(0, 1fr);
    }

    .st-key-compare-result-pair {
        grid-template-columns: minmax(0, 1fr);
    }

    .di-compare-pair-divider {
        justify-self: center;
    }

    [class*="st-key-compare-change-body-"] [data-testid="stHorizontalBlock"] {
        grid-template-columns: minmax(0, 1fr);
    }
}

/* Privacy page presentation. Keep the review-first workflow visually scoped to Privacy. */
.st-key-privacy-page {
    width: 100%;
    min-width: 0;
}

.st-key-privacy-page .di-eyebrow {
    margin-top: 0;
}

.st-key-privacy-page .di-page-title {
    margin: 0;
    color: var(--di-text);
    font-size: clamp(2rem, 3vw, 2.55rem) !important;
    font-weight: 780;
    letter-spacing: -0.055em;
    line-height: 1.08 !important;
}

.st-key-privacy-page .di-page-subtitle {
    max-width: 52rem;
    margin: 0.45rem 0 0;
    color: var(--di-muted);
    font-size: 1rem;
    line-height: 1.55;
}

.st-key-privacy-page .st-key-privacy-page-header {
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.25rem;
}

.st-key-privacy-page .st-key-privacy-page-header-copy {
    min-width: 0;
}

.di-privacy-header-icon {
    display: inline-grid;
    width: 4.15rem;
    height: 4.15rem;
    flex: 0 0 4.15rem;
    place-items: center;
    border: 1px solid #c7d2fe;
    border-radius: 1.2rem;
    background: linear-gradient(145deg, #eef2ff, #ecfeff);
    color: var(--di-indigo-dark);
    box-shadow: 0 0.65rem 1.6rem rgba(79, 70, 229, 0.1);
}

.di-privacy-header-icon svg {
    width: 2.05rem;
    height: 2.05rem;
}

.st-key-privacy-page .st-key-privacy-scan-card,
.st-key-privacy-page .st-key-privacy-overview-card,
.st-key-privacy-page .st-key-privacy-review-card,
.st-key-privacy-page .st-key-privacy-artifact-card,
.st-key-privacy-page .st-key-privacy-empty-state,
.st-key-privacy-page .st-key-privacy-no-pii-state {
    box-sizing: border-box;
    min-width: 0;
    padding: 1.35rem 1.45rem;
    border-color: #e0e4ef;
    background: #ffffff;
    box-shadow: 0 0.55rem 1.4rem rgba(17, 24, 39, 0.035);
}

.st-key-privacy-page .st-key-privacy-scan-card {
    margin-bottom: 1.2rem;
}

.st-key-privacy-page .st-key-privacy-scan-card [data-testid="stForm"],
.st-key-privacy-page .st-key-privacy-review-card [data-testid="stForm"] {
    min-width: 0;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
}

.di-privacy-card-heading,
.di-privacy-section-heading,
.di-privacy-action-heading,
.di-privacy-artifact-heading {
    display: flex;
    align-items: flex-start;
    min-width: 0;
    gap: 0.8rem;
}

.di-privacy-card-heading {
    margin-bottom: 1.15rem;
}

.di-privacy-card-icon,
.di-privacy-action-icon,
.di-privacy-artifact-icon {
    display: inline-grid;
    flex: 0 0 auto;
    place-items: center;
    border-radius: 0.8rem;
}

.di-privacy-card-icon {
    width: 2.55rem;
    height: 2.55rem;
    background: #eef2ff;
    color: var(--di-indigo-dark);
}

.di-privacy-card-icon svg,
.di-privacy-action-icon svg,
.di-privacy-artifact-icon svg {
    width: 1.35rem;
    height: 1.35rem;
}

.di-privacy-card-heading h2,
.di-privacy-section-heading h2,
.di-privacy-artifact-heading h2 {
    margin: 0;
    color: var(--di-text);
    font-size: 1.25rem;
    font-weight: 760;
    letter-spacing: -0.035em;
    line-height: 1.25;
}

.di-privacy-card-heading p,
.di-privacy-section-heading p,
.di-privacy-action-heading p,
.di-privacy-artifact-heading p {
    margin: 0.28rem 0 0;
    color: var(--di-muted);
    font-size: 0.85rem;
    line-height: 1.45;
}

.st-key-privacy-scan-fields {
    display: grid !important;
    grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr);
    align-items: end;
    gap: 1rem;
    width: 100%;
    min-width: 0;
}

.st-key-privacy-scan-fields > [data-testid="stLayoutWrapper"] {
    width: auto !important;
    min-width: 0;
}

.st-key-privacy-scan-card [data-testid="stSelectbox"] label,
.st-key-privacy-scan-card [data-testid="stMultiSelect"] label {
    color: var(--di-text);
    font-size: 0.82rem;
    font-weight: 700;
}

.st-key-privacy-scan-card [data-testid="stMultiSelect"] [data-baseweb="tag"] {
    border: 1px solid #d9dcff;
    border-radius: 999px;
    background: #f4f3ff;
    color: var(--di-indigo-dark);
}

.di-privacy-form-note {
    margin: 1rem 0 0.95rem;
    padding: 0.72rem 0.85rem;
    border: 1px solid #dbe3ff;
    border-radius: 0.7rem;
    background: #f8f9ff;
    color: var(--di-muted);
    font-size: 0.78rem;
    line-height: 1.5;
}

.di-privacy-form-note strong {
    color: var(--di-text);
}

.st-key-privacy-scan-card [data-testid="stFormSubmitButton"] button,
.st-key-privacy-review-card [data-testid="stFormSubmitButton"] button {
    min-height: 2.6rem;
    padding: 0.55rem 1rem;
    border-color: var(--di-indigo-dark);
    background: var(--di-indigo-dark);
    color: #ffffff;
    font-weight: 750;
}

.st-key-privacy-scan-card [data-testid="stFormSubmitButton"] button:hover,
.st-key-privacy-review-card [data-testid="stFormSubmitButton"] button:hover {
    border-color: #4338ca;
    background: #4338ca;
}

.di-privacy-section-kicker,
.di-privacy-artifact-kicker {
    margin-bottom: 0.55rem;
    color: var(--di-indigo-dark);
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.di-privacy-section-heading {
    margin-bottom: 1rem;
}

.st-key-privacy-overview-card {
    margin-bottom: 1.2rem;
    border-color: #dbe3ff;
    background: linear-gradient(145deg, #ffffff, #fafaff 75%, #effcff);
}

.st-key-privacy-overview-grid {
    display: grid !important;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    align-items: stretch;
    gap: 0.75rem;
    width: 100%;
    min-width: 0;
}

.st-key-privacy-overview-grid > [data-testid="stLayoutWrapper"] {
    width: auto !important;
    min-width: 0;
}

.di-privacy-metric {
    display: flex;
    min-height: 4.5rem;
    flex-direction: column;
    justify-content: center;
    gap: 0.28rem;
    padding: 0.75rem 0.8rem;
    border-left: 3px solid #c4b5fd;
    border-radius: 0.15rem;
    background: rgba(255, 255, 255, 0.78);
}

.di-privacy-metric--violet { border-left-color: #a78bfa; }
.di-privacy-metric--blue { border-left-color: #93c5fd; }
.di-privacy-metric--cyan { border-left-color: #67e8f9; }
.di-privacy-metric--green { border-left-color: #6ee7b7; }

.di-privacy-metric span {
    overflow-wrap: anywhere;
    color: var(--di-muted);
    font-size: 0.72rem;
    font-weight: 700;
    line-height: 1.25;
    text-transform: uppercase;
}

.di-privacy-metric strong {
    color: var(--di-text);
    font-size: 1.35rem;
    font-weight: 780;
    line-height: 1.1;
}

.st-key-privacy-review-card {
    margin-bottom: 1.2rem;
}

.st-key-privacy-review-card [data-testid="stWarning"] {
    margin: 0 0 0.9rem;
    border-radius: 0.7rem;
    box-shadow: none;
}

.st-key-privacy-review-card [data-testid="stForm"] > [data-testid="stVerticalBlock"] {
    gap: 0.75rem;
}

.st-key-privacy-review-card [class*="st-key-privacy-detection-"] {
    box-sizing: border-box;
    min-width: 0;
    padding: 0.8rem 0.9rem;
    border-color: #e5e7eb;
    border-radius: 0.8rem;
    background: #fcfdff;
    box-shadow: none;
}

.st-key-privacy-review-card [class*="st-key-privacy-detection-content-"] {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) auto auto;
    align-items: center;
    gap: 0.8rem;
    width: 100%;
    min-width: 0;
}

.st-key-privacy-review-card [class*="st-key-privacy-detection-content-"] > [data-testid="stLayoutWrapper"] {
    width: auto !important;
    min-width: 0;
}

.di-privacy-detection-copy {
    display: flex;
    min-width: 0;
    flex-direction: column;
    gap: 0.22rem;
}

.di-privacy-type {
    color: var(--di-indigo-dark);
    font-size: 0.67rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    line-height: 1.2;
}

.di-privacy-detection-copy strong {
    overflow-wrap: anywhere;
    color: var(--di-text);
    font-size: 0.92rem;
    font-weight: 720;
    line-height: 1.35;
}

.di-privacy-detection-copy span {
    color: var(--di-muted);
    font-size: 0.76rem;
}

.di-privacy-detection-status {
    padding: 0.27rem 0.55rem;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 750;
    white-space: nowrap;
}

.di-privacy-detection-status--eligible {
    background: #ecfdf5;
    color: #047857;
}

.di-privacy-detection-status--unavailable {
    background: #fff7ed;
    color: #c2410c;
}

.st-key-privacy-review-card [class*="st-key-privacy-detection-content-"] [data-testid="stCheckbox"] {
    min-width: max-content;
}

.st-key-privacy-review-card [data-testid="stCheckbox"] label {
    color: var(--di-text);
    font-size: 0.76rem;
    font-weight: 700;
    line-height: 1.3;
}

.st-key-privacy-redaction-action {
    box-sizing: border-box;
    min-width: 0;
    margin-top: 0.35rem;
    padding: 1rem;
    border-color: #fecdd3;
    border-radius: 0.8rem;
    background: #fffafb;
    box-shadow: none;
}

.di-privacy-action-heading {
    margin-bottom: 0.85rem;
}

.di-privacy-action-icon {
    width: 2.25rem;
    height: 2.25rem;
    background: #fff1f2;
    color: #be123c;
}

.di-privacy-action-heading h3 {
    margin: 0;
    color: var(--di-text);
    font-size: 1rem;
    font-weight: 760;
}

.st-key-privacy-redaction-action [data-testid="stFormSubmitButton"] button {
    border-color: #be123c;
    background: #be123c;
}

.st-key-privacy-redaction-action [data-testid="stFormSubmitButton"] button:hover {
    border-color: #9f1239;
    background: #9f1239;
}

.st-key-privacy-page .st-key-privacy-artifact-card {
    border-color: #a7f3d0;
    background: linear-gradient(145deg, #ffffff, #f0fdf4);
}

.di-privacy-artifact-kicker {
    color: #047857;
}

.di-privacy-artifact-icon {
    width: 2.55rem;
    height: 2.55rem;
    background: #d1fae5;
    color: #047857;
}

.st-key-privacy-artifact-card [data-testid="stDownloadButton"] button {
    margin-top: 0.95rem;
    border-color: #059669;
    background: #059669;
    color: #ffffff;
    font-weight: 750;
}

.st-key-privacy-artifact-card [data-testid="stDownloadButton"] button:hover {
    border-color: #047857;
    background: #047857;
}

.di-privacy-empty-icon {
    display: inline-grid;
    width: 2.7rem;
    height: 2.7rem;
    margin-bottom: 0.75rem;
    place-items: center;
    border-radius: 0.85rem;
    background: #eef2ff;
    color: var(--di-indigo-dark);
}

.di-privacy-empty-icon svg {
    width: 1.4rem;
    height: 1.4rem;
}

.di-privacy-empty-heading {
    margin-bottom: 0.35rem;
    color: var(--di-text);
    font-size: 1rem;
    font-weight: 750;
}

.di-privacy-empty-copy {
    max-width: 42rem;
    margin: 0;
    color: var(--di-muted);
    font-size: 0.86rem;
    line-height: 1.5;
}

.st-key-privacy-page .st-key-privacy-no-pii-state {
    border-color: #c7d2fe;
    background: #fafaff;
}

@media (max-width: 900px) {
    .st-key-privacy-overview-grid {
        grid-template-columns: repeat(3, minmax(0, 1fr));
    }

    .st-key-privacy-review-card [class*="st-key-privacy-detection-content-"] {
        grid-template-columns: minmax(0, 1fr) auto;
    }

    .st-key-privacy-review-card [class*="st-key-privacy-detection-content-"] [data-testid="stCheckbox"] {
        grid-column: 2;
        grid-row: 1 / span 2;
    }
}

@media (max-width: 560px) {
    .st-key-privacy-page .st-key-privacy-page-header {
        align-items: flex-start;
        flex-direction: column;
    }

    .st-key-privacy-page .st-key-privacy-scan-card,
    .st-key-privacy-page .st-key-privacy-overview-card,
    .st-key-privacy-page .st-key-privacy-review-card,
    .st-key-privacy-page .st-key-privacy-artifact-card,
    .st-key-privacy-page .st-key-privacy-empty-state,
    .st-key-privacy-page .st-key-privacy-no-pii-state {
        padding: 1rem;
    }

    .st-key-privacy-scan-fields,
    .st-key-privacy-overview-grid,
    .st-key-privacy-review-card [class*="st-key-privacy-detection-content-"] {
        grid-template-columns: minmax(0, 1fr);
    }

    .st-key-privacy-review-card [class*="st-key-privacy-detection-content-"] [data-testid="stCheckbox"] {
        grid-column: 1;
        grid-row: auto;
    }

    .di-privacy-detection-status {
        width: fit-content;
    }

    .st-key-privacy-page [data-testid="stFormSubmitButton"] button,
    .st-key-privacy-page [data-testid="stDownloadButton"] button {
        width: 100%;
    }
}

/* Evaluation page presentation. Keep the authoritative dashboard read-only and scoped. */
.st-key-evaluation-page {
    width: 100%;
    min-width: 0;
}

.st-key-evaluation-page .di-eyebrow {
    margin-top: 0;
}

.st-key-evaluation-page .di-page-title {
    margin: 0;
    color: var(--di-text);
    font-size: clamp(2rem, 3vw, 2.55rem) !important;
    font-weight: 780;
    letter-spacing: -0.055em;
    line-height: 1.08 !important;
}

.st-key-evaluation-page .di-page-subtitle {
    max-width: 54rem;
    margin: 0.45rem 0 0;
    color: var(--di-muted);
    font-size: 1rem;
    line-height: 1.55;
}

.st-key-evaluation-page .st-key-evaluation-page-header {
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.25rem;
}

.st-key-evaluation-page .st-key-evaluation-page-header-copy {
    min-width: 0;
}

.di-evaluation-header-icon {
    display: inline-grid;
    width: 4.15rem;
    height: 4.15rem;
    flex: 0 0 4.15rem;
    place-items: center;
    border: 1px solid #c7d2fe;
    border-radius: 1.2rem;
    background: linear-gradient(145deg, #eef2ff, #ecfeff);
    color: var(--di-indigo-dark);
    box-shadow: 0 0.65rem 1.6rem rgba(79, 70, 229, 0.1);
}

.di-evaluation-header-icon svg {
    width: 2rem;
    height: 2rem;
}

.st-key-evaluation-page .st-key-evaluation-overview-card,
.st-key-evaluation-page .st-key-evaluation-ocr-card,
.st-key-evaluation-page .st-key-evaluation-layout-card,
.st-key-evaluation-page .st-key-evaluation-retrieval-table-card,
.st-key-evaluation-page .st-key-evaluation-retrieval-latency-card,
.st-key-evaluation-page .st-key-evaluation-reranking-impact-card,
.st-key-evaluation-page .st-key-evaluation-rag-measured-card,
.st-key-evaluation-page .st-key-evaluation-rag-diagnostics-card,
.st-key-evaluation-page .st-key-evaluation-rag-citation-card,
.st-key-evaluation-page .st-key-evaluation-rag-response-format-card,
.st-key-evaluation-page .st-key-evaluation-e41-card,
.st-key-evaluation-page .st-key-evaluation-limitations-card,
.st-key-evaluation-page .st-key-evaluation-provenance-card,
.st-key-evaluation-page .st-key-evaluation-error-state {
    box-sizing: border-box;
    min-width: 0;
    padding: 1.3rem 1.4rem;
    border-color: #e0e4ef;
    background: #ffffff;
    box-shadow: 0 0.55rem 1.4rem rgba(17, 24, 39, 0.035);
}

.st-key-evaluation-page .st-key-evaluation-overview-card {
    margin-bottom: 1.2rem;
    border-color: #c7d2fe;
    background: linear-gradient(145deg, #ffffff, #fafaff 78%, #effcff);
}

.di-eval-section-heading {
    min-width: 0;
    margin-bottom: 1rem;
}

.di-eval-section-kicker,
.di-eval-subsection-title {
    color: var(--di-indigo-dark);
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.di-eval-section-heading h2 {
    margin: 0.25rem 0 0;
    color: var(--di-text);
    font-size: 1.35rem;
    font-weight: 770;
    letter-spacing: -0.04em;
    line-height: 1.2;
}

.di-eval-section-heading p {
    max-width: 58rem;
    margin: 0.35rem 0 0;
    color: var(--di-muted);
    font-size: 0.86rem;
    line-height: 1.5;
}

.st-key-evaluation-overview-grid,
.st-key-evaluation-status-grid,
.st-key-evaluation-ocr-metrics,
.st-key-evaluation-layout-metrics,
.st-key-evaluation-retrieval-quality-charts,
.st-key-evaluation-reranking-impact-grid,
.st-key-evaluation-rag-methods,
.st-key-evaluation-rag-failure-grid,
.st-key-evaluation-rag-citation-grid,
.st-key-evaluation-e41-metrics {
    display: grid !important;
    width: 100%;
    min-width: 0;
}

.st-key-evaluation-overview-grid > [data-testid="stLayoutWrapper"],
.st-key-evaluation-status-grid > [data-testid="stLayoutWrapper"],
.st-key-evaluation-ocr-metrics > [data-testid="stLayoutWrapper"],
.st-key-evaluation-layout-metrics > [data-testid="stLayoutWrapper"],
.st-key-evaluation-retrieval-quality-charts > [data-testid="stLayoutWrapper"],
.st-key-evaluation-reranking-impact-grid > [data-testid="stLayoutWrapper"],
.st-key-evaluation-rag-methods > [data-testid="stLayoutWrapper"],
.st-key-evaluation-rag-failure-grid > [data-testid="stLayoutWrapper"],
.st-key-evaluation-rag-citation-grid > [data-testid="stLayoutWrapper"],
.st-key-evaluation-e41-metrics > [data-testid="stLayoutWrapper"] {
    width: auto !important;
    min-width: 0;
}

.st-key-evaluation-overview-grid {
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.85rem;
}

.st-key-evaluation-status-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.7rem;
    margin-top: 0.85rem;
}

.st-key-evaluation-overview-grid [data-testid="stVerticalBlockBorderWrapper"],
.st-key-evaluation-status-grid [data-testid="stVerticalBlockBorderWrapper"] {
    height: 100%;
    box-sizing: border-box;
    border-color: #e0e4ef;
    background: rgba(255, 255, 255, 0.82);
    box-shadow: none;
}

.di-eval-overview-value,
.di-eval-status-value {
    display: flex;
    min-height: 4.25rem;
    flex-direction: column;
    justify-content: center;
    gap: 0.3rem;
    padding: 0.75rem 0.85rem;
    border-left: 3px solid #c4b5fd;
}

.di-eval-overview-value--cyan { border-left-color: #67e8f9; }
.di-eval-overview-value--violet { border-left-color: #a78bfa; }
.di-eval-overview-value--green { border-left-color: #6ee7b7; }

.di-eval-overview-value span,
.di-eval-status-value span {
    overflow-wrap: anywhere;
    color: var(--di-muted);
    font-size: 0.72rem;
    font-weight: 750;
    line-height: 1.25;
    text-transform: uppercase;
}

.di-eval-overview-value strong {
    color: var(--di-text);
    font-size: 1.4rem;
    font-weight: 780;
    line-height: 1.1;
}

.di-eval-status-value {
    min-height: 3.25rem;
    flex-direction: row;
    align-items: center;
    justify-content: space-between;
    border-left-width: 2px;
}

.di-eval-status-value--green { border-left-color: #6ee7b7; }
.di-eval-status-value--amber { border-left-color: #fbbf24; }
.di-eval-status-value--neutral { border-left-color: #cbd5e1; }

.di-eval-status-value strong {
    color: var(--di-text);
    font-size: 1.1rem;
    font-weight: 780;
}

.st-key-evaluation-ocr-card,
.st-key-evaluation-layout-card {
    margin-bottom: 1.15rem;
}

.di-eval-subsection-title {
    margin-bottom: 0.8rem;
}

.st-key-evaluation-ocr-metrics {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.85rem;
}

.st-key-evaluation-layout-metrics {
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 0.7rem;
}

.st-key-evaluation-page [data-testid="stMetric"] {
    min-width: 0;
}

.st-key-evaluation-page [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMetricValue"] {
    overflow-wrap: anywhere;
}

.di-eval-state {
    display: inline-flex;
    width: fit-content;
    margin: 0.2rem 0 0.45rem;
    padding: 0.2rem 0.5rem;
    border-radius: 999px;
    font-size: 0.67rem;
    font-weight: 800;
    letter-spacing: 0.04em;
}

.di-eval-state--blocked {
    background: #fff7ed;
    color: #c2410c;
}

.di-eval-state--not_applicable,
.di-eval-state--not_measured {
    background: #f1f5f9;
    color: #475569;
}

.di-eval-interpretation {
    margin-top: 0.95rem;
    padding: 0.8rem 0.9rem;
    border: 1px solid #dbe3ff;
    border-radius: 0.7rem;
    background: #f8f9ff;
    color: var(--di-muted);
    font-size: 0.8rem;
    line-height: 1.5;
}

.di-eval-interpretation strong {
    color: var(--di-text);
}

.st-key-evaluation-page .st-key-evaluation-retrieval-table-card {
    margin-bottom: 1.2rem;
    padding-bottom: 1rem;
}

.st-key-evaluation-retrieval-table-card [data-testid="stDataFrame"] {
    width: 100%;
    max-width: 100%;
}

.st-key-evaluation-retrieval-table-card iframe {
    max-width: 100%;
}

.st-key-evaluation-retrieval-quality-charts {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.9rem;
    margin-bottom: 1.25rem;
}

.st-key-evaluation-retrieval-quality-charts [data-testid="stVerticalBlockBorderWrapper"],
.st-key-evaluation-retrieval-latency-card [data-testid="stVerticalBlockBorderWrapper"] {
    box-sizing: border-box;
    min-width: 0;
    border-color: #e0e4ef;
    background: #ffffff;
    box-shadow: none;
}

.di-eval-chart-title,
.di-eval-method-label {
    margin-bottom: 0.55rem;
    color: var(--di-text);
    font-size: 0.88rem;
    font-weight: 760;
}

.st-key-evaluation-retrieval-latency-card {
    margin-bottom: 1.2rem;
    border-color: #e0e7ff;
    background: #fbfcff;
}

.st-key-evaluation-reranking-impact-card {
    margin-bottom: 1.2rem;
    border-color: #c7d2fe;
    background: linear-gradient(145deg, #ffffff, #fafaff 68%, #effcff);
}

.st-key-evaluation-reranking-impact-grid {
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.7rem;
}

.st-key-evaluation-reranking-impact-grid [data-testid="stVerticalBlockBorderWrapper"] {
    height: 100%;
    box-sizing: border-box;
    border-color: #e0e4ef;
    background: rgba(255, 255, 255, 0.86);
    box-shadow: none;
}

.di-eval-impact-value {
    display: flex;
    min-height: 4rem;
    flex-direction: column;
    justify-content: center;
    gap: 0.28rem;
    padding: 0.72rem 0.8rem;
    border-left: 3px solid #a5b4fc;
}

.di-eval-impact-value--cyan { border-left-color: #67e8f9; }
.di-eval-impact-value--green { border-left-color: #6ee7b7; }
.di-eval-impact-value--amber { border-left-color: #fbbf24; }

.di-eval-impact-value span {
    color: var(--di-muted);
    font-size: 0.7rem;
    font-weight: 750;
    line-height: 1.25;
}

.di-eval-impact-value strong {
    color: var(--di-text);
    font-size: 1.2rem;
    font-weight: 780;
}

.st-key-evaluation-rag-measured-card,
.st-key-evaluation-rag-diagnostics-card,
.st-key-evaluation-rag-citation-card,
.st-key-evaluation-rag-response-format-card,
.st-key-evaluation-e41-card {
    margin-bottom: 1.15rem;
}

.st-key-evaluation-rag-measured-card [data-testid="stAlert"] {
    margin-bottom: 1rem;
    border-radius: 0.75rem;
    box-shadow: none;
}

.st-key-evaluation-rag-methods {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.9rem;
}

.st-key-evaluation-rag-methods > [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlockBorderWrapper"],
.st-key-evaluation-rag-citation-grid > [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlockBorderWrapper"] {
    height: 100%;
    box-sizing: border-box;
    border-color: #e0e4ef;
    background: #fcfdff;
    box-shadow: none;
}

.st-key-evaluation-rag-hybrid-metrics,
.st-key-evaluation-rag-hybrid-reranked-metrics {
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.55rem;
}

.st-key-evaluation-rag-hybrid-quality,
.st-key-evaluation-rag-hybrid-reranked-quality {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.55rem;
    margin-top: 0.55rem;
}

.st-key-evaluation-rag-failure-grid {
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.65rem;
}

.st-key-evaluation-rag-citation-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.9rem;
}

.st-key-evaluation-rag-citation-grid .di-eval-method-label {
    color: var(--di-indigo-dark);
}

.st-key-evaluation-e41-metrics {
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.7rem;
}

.st-key-evaluation-rag-response-format-card {
    border-color: #fde68a;
    background: #fffdf5;
}

.st-key-evaluation-e41-card {
    border-color: #fed7aa;
    background: #fffaf5;
}

.st-key-evaluation-page [data-testid="stTabs"] [role="tablist"] {
    gap: 1.25rem;
    border-bottom: 1px solid #e0e4ef;
}

.st-key-evaluation-page [data-testid="stTabs"] [role="tab"] {
    color: var(--di-muted);
    font-size: 0.82rem;
    font-weight: 700;
}

.st-key-evaluation-page [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--di-indigo-dark);
}

.st-key-evaluation-page .st-key-evaluation-limitations-card [data-testid="stExpander"] {
    margin: 0.35rem 0;
    border-color: #e0e4ef;
    border-radius: 0.75rem;
    background: #fcfdff;
    box-shadow: none;
}

.st-key-evaluation-page .st-key-evaluation-limitations-card [data-testid="stExpander"] summary {
    color: var(--di-text);
    font-size: 0.84rem;
    font-weight: 720;
}

.di-eval-provenance-row {
    display: grid;
    grid-template-columns: minmax(7rem, 0.35fr) minmax(0, 1fr);
    gap: 0.8rem;
    padding: 0.72rem 0;
    border-bottom: 1px solid #eef0f5;
}

.di-eval-provenance-row:last-child {
    border-bottom: 0;
}

.di-eval-provenance-row span {
    color: var(--di-muted);
    font-size: 0.78rem;
    font-weight: 700;
}

.di-eval-provenance-row strong {
    overflow-wrap: anywhere;
    color: var(--di-text);
    font-size: 0.8rem;
    font-weight: 650;
}

@media (max-width: 900px) {
    .st-key-evaluation-overview-grid,
    .st-key-evaluation-layout-metrics,
    .st-key-evaluation-reranking-impact-grid,
    .st-key-evaluation-e41-metrics {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .st-key-evaluation-rag-failure-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 560px) {
    .st-key-evaluation-page .st-key-evaluation-page-header {
        align-items: flex-start;
        flex-direction: column;
    }

    .st-key-evaluation-page .st-key-evaluation-overview-card,
    .st-key-evaluation-page .st-key-evaluation-ocr-card,
    .st-key-evaluation-page .st-key-evaluation-layout-card,
    .st-key-evaluation-page .st-key-evaluation-retrieval-table-card,
    .st-key-evaluation-page .st-key-evaluation-retrieval-latency-card,
    .st-key-evaluation-page .st-key-evaluation-reranking-impact-card,
    .st-key-evaluation-page .st-key-evaluation-rag-measured-card,
    .st-key-evaluation-page .st-key-evaluation-rag-diagnostics-card,
    .st-key-evaluation-page .st-key-evaluation-rag-citation-card,
    .st-key-evaluation-page .st-key-evaluation-rag-response-format-card,
    .st-key-evaluation-page .st-key-evaluation-e41-card,
    .st-key-evaluation-page .st-key-evaluation-limitations-card,
    .st-key-evaluation-page .st-key-evaluation-provenance-card {
        padding: 1rem;
    }

    .st-key-evaluation-overview-grid,
    .st-key-evaluation-status-grid,
    .st-key-evaluation-ocr-metrics,
    .st-key-evaluation-layout-metrics,
    .st-key-evaluation-retrieval-quality-charts,
    .st-key-evaluation-reranking-impact-grid,
    .st-key-evaluation-rag-methods,
    .st-key-evaluation-rag-failure-grid,
    .st-key-evaluation-rag-citation-grid,
    .st-key-evaluation-e41-metrics,
    .st-key-evaluation-rag-hybrid-metrics,
    .st-key-evaluation-rag-hybrid-reranked-metrics,
    .st-key-evaluation-rag-hybrid-quality,
    .st-key-evaluation-rag-hybrid-reranked-quality {
        grid-template-columns: minmax(0, 1fr);
    }

    .di-eval-provenance-row {
        grid-template-columns: minmax(0, 1fr);
        gap: 0.25rem;
    }

    .st-key-evaluation-page [data-testid="stTabs"] [role="tablist"] {
        gap: 0.85rem;
        overflow-x: auto;
    }
}

/* Ask page presentation. This scope keeps the accepted Home and Documents layouts stable. */
.st-key-ask-page {
    width: 100%;
    min-width: 0;
}

.st-key-ask-page .di-eyebrow {
    margin-top: 0;
}

.st-key-ask-page .di-page-title {
    margin: 0;
    color: var(--di-text);
    font-size: clamp(2rem, 3vw, 2.55rem) !important;
    font-weight: 780;
    letter-spacing: -0.055em;
    line-height: 1.08 !important;
}

.st-key-ask-page .di-page-subtitle {
    max-width: 52rem;
    margin: 0.45rem 0 0;
    color: var(--di-muted);
    font-size: 1rem;
    line-height: 1.55;
}

.st-key-ask-page .st-key-ask-page-header {
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.25rem;
}

.st-key-ask-page .st-key-ask-page-header-copy {
    min-width: 0;
}

.di-ask-header-icon {
    display: inline-grid;
    width: 4.15rem;
    height: 4.15rem;
    flex: 0 0 4.15rem;
    place-items: center;
    border: 1px solid #c7d2fe;
    border-radius: 1.2rem;
    background: linear-gradient(145deg, #eef2ff, #ecfeff);
    color: var(--di-indigo-dark);
    box-shadow: 0 0.65rem 1.6rem rgba(79, 70, 229, 0.1);
}

.di-ask-header-icon svg {
    width: 2rem;
    height: 2rem;
}

.st-key-ask-page [data-testid="stTabs"] [role="tablist"] {
    gap: 1.45rem;
    border-bottom: 1px solid var(--di-border);
}

.st-key-ask-page [data-testid="stTabs"] button[role="tab"] {
    min-height: 2.85rem;
    padding: 0.25rem 0.05rem 0.7rem;
    border-bottom: 2px solid transparent;
    color: #4b5563;
    font-size: 0.88rem;
    font-weight: 650;
}

.st-key-ask-page [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    border-bottom-color: var(--di-indigo);
    color: var(--di-indigo-dark);
    font-weight: 760;
}

.di-ask-section-heading,
.di-evidence-section-heading,
.di-results-heading {
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
    margin: 1.6rem 0 0.9rem;
}

.di-ask-section-icon,
.di-evidence-section-icon,
.di-results-count {
    display: inline-grid;
    width: 2.3rem;
    height: 2.3rem;
    flex: 0 0 2.3rem;
    place-items: center;
    border-radius: 0.75rem;
    background: #eef2ff;
    color: var(--di-indigo-dark);
}

.di-ask-section-icon svg,
.di-evidence-section-icon svg {
    width: 1.25rem;
    height: 1.25rem;
}

.di-ask-section-heading h2,
.di-evidence-section-heading h2,
.di-results-heading h2 {
    margin: 0;
    color: var(--di-text);
    font-size: 1.3rem;
    font-weight: 760;
    letter-spacing: -0.035em;
    line-height: 1.25;
}

.di-ask-section-heading p,
.di-evidence-section-heading p,
.di-results-heading p {
    margin: 0.3rem 0 0;
    color: var(--di-muted);
    font-size: 0.86rem;
    line-height: 1.45;
}

.st-key-ask-grounded-card,
.st-key-ask-search-card,
.st-key-ask-conversations-card,
.st-key-ask-answer-card {
    min-width: 0;
    box-sizing: border-box;
    padding: 1.25rem 1.35rem;
}

.st-key-ask-grounded-card,
.st-key-ask-search-card,
.st-key-ask-conversations-card {
    border-color: #e0e4ef;
    background: #ffffff;
    box-shadow: 0 0.55rem 1.4rem rgba(17, 24, 39, 0.035);
}

.st-key-ask-grounded-card [data-testid="stVerticalBlockBorderWrapper"],
.st-key-ask-search-card [data-testid="stVerticalBlockBorderWrapper"],
.st-key-ask-conversations-card [data-testid="stVerticalBlockBorderWrapper"],
.st-key-ask-answer-card [data-testid="stVerticalBlockBorderWrapper"] {
    min-width: 0;
    padding: 1.25rem 1.35rem;
}

.st-key-ask-grounded-card [data-testid="stForm"],
.st-key-ask-search-card [data-testid="stForm"],
.st-key-ask-conversations-card [data-testid="stForm"] {
    min-width: 0;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
}

.di-form-section-label {
    margin: 0.15rem 0 0.5rem;
    color: var(--di-text);
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.di-form-section-label--settings {
    margin-top: 1.15rem;
}

.st-key-ask-retrieval-settings,
.st-key-search-retrieval-settings {
    display: grid !important;
    grid-template-columns: minmax(0, 1.2fr) minmax(7rem, 0.6fr) minmax(0, 1fr);
    align-items: end;
    gap: 0.9rem;
    width: 100%;
    min-width: 0;
}

.st-key-ask-retrieval-settings > [data-testid="stLayoutWrapper"],
.st-key-search-retrieval-settings > [data-testid="stLayoutWrapper"] {
    width: auto !important;
    min-width: 0;
}

.st-key-ask-filter-panel,
.st-key-search-filter-panel,
[class*="st-key-conversation-"][class*="-filter-panel"] {
    margin-top: 1rem;
}

.st-key-ask-filter-panel [data-testid="stVerticalBlockBorderWrapper"],
.st-key-search-filter-panel [data-testid="stVerticalBlockBorderWrapper"],
[class*="st-key-conversation-"][class*="-filter-panel"] [data-testid="stVerticalBlockBorderWrapper"] {
    padding: 1rem 1.05rem;
    border-color: #e0e4ef;
    background: #fbfcff;
    box-shadow: none;
}

.st-key-ask-filter-panel,
.st-key-search-filter-panel,
[class*="st-key-conversation-"][class*="-filter-panel"] {
    box-sizing: border-box;
    padding: 1rem 1.05rem;
    border-color: #e0e4ef;
    background: #fbfcff;
    box-shadow: none;
}

.di-filter-heading {
    display: flex;
    align-items: flex-start;
    gap: 0.55rem;
    margin-bottom: 0.75rem;
}

.di-filter-icon {
    display: inline-grid;
    width: 1.65rem;
    height: 1.65rem;
    flex: 0 0 1.65rem;
    place-items: center;
    border-radius: 0.5rem;
    background: #eef2ff;
    color: var(--di-indigo-dark);
}

.di-filter-icon svg {
    width: 1rem;
    height: 1rem;
}

.di-filter-heading strong,
.di-filter-heading span {
    display: block;
}

.di-filter-heading strong {
    color: var(--di-text);
    font-size: 0.86rem;
    font-weight: 750;
}

.di-filter-heading span {
    margin-top: 0.15rem;
    color: var(--di-muted);
    font-size: 0.75rem;
}

.st-key-ask-grounded-card [data-testid="stTextArea"] textarea {
    min-height: 8.75rem;
}

.st-key-ask-grounded-card [data-testid="stFormSubmitButton"] button,
.st-key-ask-search-card [data-testid="stFormSubmitButton"] button,
.st-key-ask-conversations-card [data-testid="stFormSubmitButton"] button {
    min-height: 2.55rem;
}

.st-key-ask-answer-card {
    margin-top: 1.35rem;
}

.st-key-ask-answer-card [data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #c7d2fe;
    background: linear-gradient(145deg, #ffffff, #f7f8ff 72%, #effcff);
    box-shadow: 0 0.8rem 2rem rgba(79, 70, 229, 0.07);
}

.st-key-ask-answer-card {
    border-color: #c7d2fe;
    background: linear-gradient(145deg, #ffffff, #f7f8ff 72%, #effcff);
    box-shadow: 0 0.8rem 2rem rgba(79, 70, 229, 0.07);
}

.di-answer-heading {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-bottom: 0.9rem;
}

.di-answer-icon {
    display: inline-grid;
    width: 2.25rem;
    height: 2.25rem;
    flex: 0 0 2.25rem;
    place-items: center;
    border-radius: 0.7rem;
    background: #eef2ff;
    color: var(--di-indigo-dark);
}

.di-answer-icon svg {
    width: 1.25rem;
    height: 1.25rem;
}

.di-answer-kicker,
.di-evidence-label {
    color: var(--di-indigo-dark);
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.di-answer-heading h2 {
    margin: 0.12rem 0 0;
    color: var(--di-text);
    font-size: 1.15rem;
    font-weight: 760;
}

.st-key-ask-page [data-testid="stExpander"] {
    margin: 0.7rem 0;
    border-color: #e0e4ef;
    border-radius: 0.85rem;
    background: rgba(255, 255, 255, 0.72);
}

.st-key-ask-page [data-testid="stExpander"] summary {
    color: var(--di-text);
    font-size: 0.86rem;
    font-weight: 700;
}

.di-evidence-card-heading {
    display: flex;
    align-items: flex-start;
    gap: 0.7rem;
    margin-bottom: 0.8rem;
}

.di-source-badge {
    display: inline-grid;
    min-width: 2.35rem;
    height: 2.1rem;
    padding: 0 0.45rem;
    place-items: center;
    box-sizing: border-box;
    border-radius: 0.65rem;
    background: #eef2ff;
    color: var(--di-indigo-dark);
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.04em;
}

.di-evidence-card-heading strong,
.di-evidence-card-heading span {
    display: block;
    overflow-wrap: anywhere;
}

.di-evidence-card-heading strong {
    color: var(--di-text);
    font-size: 0.91rem;
    font-weight: 750;
}

.di-evidence-card-heading span {
    margin-top: 0.2rem;
    color: var(--di-muted);
    font-size: 0.78rem;
}

.di-evidence-label {
    margin-bottom: 0.3rem;
    color: var(--di-muted);
    letter-spacing: 0.08em;
}

.di-evidence-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem 1.1rem;
    margin-top: 0.8rem;
    padding-top: 0.7rem;
    border-top: 1px solid #edf0f6;
    color: var(--di-muted);
    font-size: 0.73rem;
    line-height: 1.4;
}

.di-evidence-meta strong {
    color: #374151;
    font-weight: 700;
}

.di-results-count {
    background: #ecfeff;
    color: #0e7490;
    font-size: 0.82rem;
    font-weight: 800;
}

.st-key-ask-conversations-card [data-testid="stButton"] {
    margin-bottom: 0.45rem;
}

.st-key-conversation-history [data-testid="stVerticalBlockBorderWrapper"],
.st-key-conversation-controls [data-testid="stVerticalBlockBorderWrapper"] {
    padding: 1rem 1.05rem;
    border-color: #e0e4ef;
    background: #fbfcff;
    box-shadow: none;
}

.st-key-conversation-history,
.st-key-conversation-controls {
    box-sizing: border-box;
    padding: 1rem 1.05rem;
    border-color: #e0e4ef;
    background: #fbfcff;
    box-shadow: none;
}

.st-key-conversation-history {
    margin-top: 1.1rem;
}

.di-chat-role {
    margin-bottom: 0.3rem;
    color: var(--di-indigo-dark);
    font-size: 0.66rem;
    font-weight: 800;
    letter-spacing: 0.13em;
}

.st-key-conversation-history [data-testid="stChatMessage"] {
    margin: 0.55rem 0;
    padding: 0.75rem 0.9rem;
    border: 1px solid #e5e7eb;
    border-radius: 0.8rem;
    background: #ffffff;
}

.st-key-conversation-history [data-testid="stChatMessage"]:has(.di-chat-role) {
    box-shadow: none;
}

.st-key-conversation-controls {
    margin-top: 1rem;
}

.st-key-ask-conversations-card [data-testid="stChatInput"] {
    margin-top: 0.85rem;
}

.st-key-conversation-danger-zone {
    margin-top: 1rem;
}

.st-key-conversation-danger-zone [data-testid="stVerticalBlockBorderWrapper"] {
    padding: 0.95rem 1.05rem;
    border-color: #fecaca;
    background: #fffafa;
    box-shadow: none;
}

.st-key-conversation-danger-zone {
    box-sizing: border-box;
    padding: 0.95rem 1.05rem;
    border-color: #fecaca;
    background: #fffafa;
    box-shadow: none;
}

.di-danger-heading {
    color: #991b1b;
    font-size: 0.86rem;
    font-weight: 750;
}

.st-key-conversation-danger-zone [data-testid="stForm"] {
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
}

.st-key-conversation-danger-zone [data-testid="stFormSubmitButton"] button {
    border-color: #fecaca;
    color: #b91c1c;
}

.st-key-conversation-danger-zone [data-testid="stFormSubmitButton"] button:hover {
    border-color: #ef4444;
    background: #fef2f2;
}

.st-key-home-kpi-grid,
.st-key-home-capabilities-grid {
    display: grid !important;
    width: 100%;
    min-width: 0;
    align-items: stretch;
    gap: 1rem;
}

.st-key-home-kpi-grid {
    grid-template-columns: repeat(4, minmax(0, 1fr));
    margin-top: 0.4rem;
}

.st-key-home-capabilities-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr));
}

.st-key-home-kpi-grid > div,
.st-key-home-capabilities-grid > div {
    width: auto !important;
    min-width: 0;
}

.st-key-home-capabilities-grid > div {
    min-height: 7.4rem;
}

.st-key-home-capabilities-grid > [data-testid="stLayoutWrapper"] {
    min-height: 7.4rem !important;
}

.st-key-home-kpi-grid [data-testid="stVerticalBlockBorderWrapper"],
.st-key-home-capabilities-grid [data-testid="stVerticalBlockBorderWrapper"] {
    height: 100%;
    box-sizing: border-box;
}

@media (max-width: 900px) {
    [data-testid="stAppViewContainer"] > section[data-testid="stSidebar"] + div {
        width: 100% !important;
        margin-left: 0 !important;
        flex: 1 1 auto !important;
    }

    [data-testid="stMainBlockContainer"] {
        padding-top: 1.25rem;
    }

    .st-key-home-hero-grid {
        grid-template-columns: minmax(0, 1fr);
    }

    .st-key-home-kpi-grid,
    .st-key-home-capabilities-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .di-pipeline-card {
        min-height: auto;
        margin-top: 1rem;
    }

    .st-key-documents-page .st-key-documents-page-header {
        align-items: flex-start;
        flex-direction: column;
    }

    .st-key-document-overview-kpis {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .st-key-document-actions {
        align-items: stretch;
        flex-direction: column;
    }

    .st-key-document-actions [data-testid="stForm"] {
        align-items: flex-start;
        flex-wrap: wrap;
    }

    .st-key-ask-page .st-key-ask-page-header {
        align-items: flex-start;
    }

    .st-key-analyze-page .st-key-analyze-page-header {
        align-items: flex-start;
    }

    .st-key-analyze-document-kpis {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .st-key-ask-retrieval-settings,
    .st-key-search-retrieval-settings {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 560px) {
    .st-key-home-kpi-grid,
    .st-key-home-capabilities-grid {
        grid-template-columns: minmax(0, 1fr);
    }

    .di-pipeline-main-row,
    .di-pipeline-rail {
        grid-template-columns: 1fr;
    }

    .di-pipeline-rail--top .di-pipeline-source,
    .di-pipeline-rail--bottom .di-pipeline-source,
    .di-pipeline-rail--top .di-pipeline-rail-arrow,
    .di-pipeline-rail--bottom .di-pipeline-rail-arrow {
        grid-column: 1;
    }

    .di-pipeline-connector {
        transform: rotate(90deg);
    }

    .di-pipeline-source {
        width: min(100%, 10rem);
    }

    .st-key-document-overview-kpis {
        grid-template-columns: minmax(0, 1fr);
    }

    .st-key-document-actions [data-testid="stForm"] {
        align-items: stretch;
        flex-direction: column;
    }

    .st-key-ask-page .st-key-ask-page-header {
        align-items: flex-start;
        flex-direction: column;
    }

    .st-key-analyze-page .st-key-analyze-page-header {
        align-items: flex-start;
        flex-direction: column;
    }

    .st-key-analyze-page [data-testid="stTabs"] [role="tablist"] {
        gap: 0.85rem;
        overflow-x: auto;
    }

    .st-key-analyze-document-kpis {
        grid-template-columns: minmax(0, 1fr);
    }

    .st-key-analyze-page .st-key-analyze-document-context,
    .st-key-analyze-page .st-key-analyze-summary-card,
    .st-key-analyze-page .st-key-analyze-classification-card,
    .st-key-analyze-page .st-key-analyze-extraction-card,
    .st-key-analyze-page .st-key-analyze-tables-card {
        padding: 1rem;
    }

    .st-key-ask-page [data-testid="stTabs"] [role="tablist"] {
        gap: 0.85rem;
        overflow-x: auto;
    }

    .st-key-ask-retrieval-settings,
    .st-key-search-retrieval-settings {
        grid-template-columns: minmax(0, 1fr);
    }

    .st-key-ask-page [class*="filter-panel"] [data-testid="stHorizontalBlock"] {
        display: grid !important;
        grid-template-columns: minmax(0, 1fr) !important;
        gap: 0.65rem;
    }

    .st-key-ask-page [class*="filter-panel"] [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        width: auto !important;
        min-width: 0 !important;
    }

    .st-key-ask-grounded-card [data-testid="stVerticalBlockBorderWrapper"],
    .st-key-ask-search-card [data-testid="stVerticalBlockBorderWrapper"],
    .st-key-ask-conversations-card [data-testid="stVerticalBlockBorderWrapper"],
    .st-key-ask-answer-card [data-testid="stVerticalBlockBorderWrapper"] {
        padding: 1rem;
    }

    .st-key-ask-grounded-card,
    .st-key-ask-search-card,
    .st-key-ask-conversations-card,
    .st-key-ask-answer-card {
        padding: 1rem;
    }

    .di-page-view-heading {
        align-items: flex-start;
        flex-direction: column;
        gap: 0.3rem;
    }
}
</style>
"""


def apply_visual_system() -> None:
    """Apply the shared DocuIntel visual system to the current app run."""

    st.markdown(VISUAL_SYSTEM_CSS, unsafe_allow_html=True)


def render_sidebar_brand() -> None:
    """Render the compact sidebar brand block."""

    st.sidebar.markdown(
        """
        <div class="di-brand">
            <div class="di-brand-lockup">
                <div class="di-brand-mark" aria-hidden="true">
                    <svg viewBox="0 0 42 42" role="img" aria-label="DocuIntel mark">
                        <defs>
                            <linearGradient id="di-brand-gradient" x1="7" y1="5" x2="35" y2="37" gradientUnits="userSpaceOnUse">
                                <stop stop-color="#38bdf8"/>
                                <stop offset="0.52" stop-color="#6366f1"/>
                                <stop offset="1" stop-color="#a855f7"/>
                            </linearGradient>
                        </defs>
                        <path d="M21 3.5 36 12 21 20.7 6 12 21 3.5Z" fill="#38bdf8" fill-opacity="0.2"/>
                        <path d="M6 12 21 20.7v17L6 30V12Z" fill="#6366f1" fill-opacity="0.2"/>
                        <path d="m21 20.7 15-8v17l-15 8.8v-17Z" fill="#a855f7" fill-opacity="0.2"/>
                        <path d="M21 3.5 36 12v18L21 38.5 6 30V12L21 3.5Z" fill="none" stroke="url(#di-brand-gradient)" stroke-width="3.6" stroke-linejoin="round"/>
                        <path d="m7.1 12.7 13.9 8 13.9-8M21 20.7v17" fill="none" stroke="url(#di-brand-gradient)" stroke-width="3.1" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="m13.2 8.1 7.8 4.5 7.8-4.5" fill="none" stroke="#38bdf8" stroke-width="2.1" stroke-linecap="round"/>
                    </svg>
                </div>
                <div>
                    <div class="di-brand-name">DocuIntel</div>
                    <div class="di-brand-subtitle">Document AI platform</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_visual() -> None:
    """Render a self-contained, local connected representation of the pipeline."""

    st.markdown(
        """
        <div class="di-pipeline-card" aria-label="DocuIntel document intelligence pipeline">
            <div class="di-pipeline-diagram">
                <div class="di-pipeline-rail di-pipeline-rail--top">
                    <div class="di-pipeline-source">
                        <span class="di-pipeline-source-icon">∿</span>
                        <span><strong>Embeddings<br>&amp; Vectors</strong><small>semantic context</small></span>
                    </div>
                </div>
                <div class="di-pipeline-rail di-pipeline-rail--top">
                    <span class="di-pipeline-rail-arrow" aria-hidden="true">↓</span>
                </div>
                <div class="di-pipeline-main-row">
                    <div class="di-document-stack" aria-label="PDF document">
                        <span class="di-document-sheet di-document-sheet--back"></span>
                        <span class="di-document-sheet di-document-sheet--middle"></span>
                        <span class="di-document-sheet di-document-sheet--front">
                            <span class="di-pdf-badge">PDF</span>
                            <span class="di-document-lines"></span>
                            <span class="di-document-art"></span>
                        </span>
                    </div>
                    <span class="di-pipeline-connector" aria-hidden="true">→</span>
                    <div class="di-pipeline-node">
                        <span class="di-pipeline-icon">⌁</span>
                        <strong>OCR &amp; Extraction</strong>
                        <small>text + layout</small>
                    </div>
                    <span class="di-pipeline-connector" aria-hidden="true">→</span>
                    <div class="di-pipeline-node">
                        <span class="di-pipeline-icon">▤</span>
                        <strong>Structure &amp; Tables</strong>
                        <small>document understanding</small>
                    </div>
                    <span class="di-pipeline-connector" aria-hidden="true">→</span>
                    <div class="di-pipeline-core">
                        <svg class="di-brain-svg" viewBox="0 0 100 100" aria-hidden="true">
                            <path d="M50 20c-6-8-18-7-22 3-9-2-16 6-13 15-8 5-7 17 1 21-4 9 3 19 13 18 4 9 16 9 21 2" fill="none" stroke="#dbeafe" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
                            <path d="M50 20c6-8 18-7 22 3 9-2 16 6 13 15 8 5 7 17-1 21 4 9-3 19-13 18-4 9-16 9-21 2" fill="none" stroke="#c4b5fd" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
                            <path d="M50 21v58M29 36c7 1 10 5 10 11M71 36c-7 1-10 5-10 11M28 57c8-2 12 2 12 9M72 57c-8-2-12 2-12 9M40 30c3 4 3 8 0 12M60 30c-3 4-3 8 0 12" fill="none" stroke="#ffffff" stroke-width="2.2" stroke-linecap="round"/>
                            <circle cx="50" cy="21" r="3" fill="#67e8f9"/>
                            <circle cx="39" cy="47" r="2.5" fill="#a5b4fc"/>
                            <circle cx="61" cy="47" r="2.5" fill="#a5b4fc"/>
                            <circle cx="50" cy="79" r="3" fill="#67e8f9"/>
                        </svg>
                        <span class="di-core-label">AI intelligence core</span>
                    </div>
                    <span class="di-pipeline-connector" aria-hidden="true">→</span>
                    <div class="di-pipeline-node">
                        <span class="di-pipeline-icon">⌕</span>
                        <strong>Hybrid Retrieval</strong>
                        <small>keyword + semantic</small>
                    </div>
                    <span class="di-pipeline-connector" aria-hidden="true">→</span>
                    <div class="di-pipeline-node">
                        <span class="di-pipeline-icon">♜</span>
                        <strong>Reranker</strong>
                        <small>CrossEncoder precision</small>
                    </div>
                    <span class="di-pipeline-connector" aria-hidden="true">→</span>
                    <div class="di-pipeline-node">
                        <span class="di-pipeline-icon">✓</span>
                        <strong>Grounded Answer</strong>
                        <small>citations through Ollama</small>
                    </div>
                </div>
                <div class="di-pipeline-rail di-pipeline-rail--bottom">
                    <span class="di-pipeline-rail-arrow" aria-hidden="true">↑</span>
                </div>
                <div class="di-pipeline-rail di-pipeline-rail--bottom">
                    <div class="di-pipeline-source di-pipeline-source--database">
                        <span class="di-pipeline-source-icon">▤</span>
                        <span><strong>PostgreSQL<br>+ pgvector</strong><small>persisted knowledge store</small></span>
                    </div>
                </div>
            </div>
            <div class="di-pipeline-footer">Every answer stays connected to retrieved document evidence.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


__all__ = ["apply_visual_system", "render_pipeline_visual", "render_sidebar_brand"]
