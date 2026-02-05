# =============================================================================
# IKON SVG UNTUK SIDEBAR
# CyberGuard AI: Enterprise Edition
# =============================================================================

# Ikon menggunakan SVG inline untuk tampilan yang lebih profesional
ICONS = {
    "dashboard": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#00ffff" stroke-width="2">
        <rect x="3" y="3" width="7" height="7" rx="1"/>
        <rect x="14" y="3" width="7" height="7" rx="1"/>
        <rect x="3" y="14" width="7" height="7" rx="1"/>
        <rect x="14" y="14" width="7" height="7" rx="1"/>
    </svg>""",
    
    "simulation": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#00ffff" stroke-width="2">
        <circle cx="11" cy="11" r="8"/>
        <path d="M21 21l-4.35-4.35"/>
    </svg>""",
    
    "batch": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#00ffff" stroke-width="2">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
        <polyline points="14 2 14 8 20 8"/>
        <line x1="16" y1="13" x2="8" y2="13"/>
        <line x1="16" y1="17" x2="8" y2="17"/>
    </svg>""",
    
    "performance": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#00ffff" stroke-width="2">
        <path d="M12 2a10 10 0 1 0 10 10H12V2z"/>
        <path d="M12 2a10 10 0 0 1 10 10"/>
        <circle cx="12" cy="12" r="3"/>
    </svg>""",
    
    "shield": """<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#00ffff" stroke-width="2">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
        <path d="M9 12l2 2 4-4" stroke="#00ff88"/>
    </svg>""",
    
    "status": """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#00ffff" stroke-width="2">
        <circle cx="12" cy="12" r="10"/>
        <polyline points="12 6 12 12 16 14"/>
    </svg>""",
    
    "settings": """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#00ffff" stroke-width="2">
        <circle cx="12" cy="12" r="3"/>
        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>
    </svg>""",
    
    "language": """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#00ffff" stroke-width="2">
        <circle cx="12" cy="12" r="10"/>
        <line x1="2" y1="12" x2="22" y2="12"/>
        <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
    </svg>""",
    
    "chart": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#00ffff" stroke-width="2">
        <line x1="18" y1="20" x2="18" y2="10"/>
        <line x1="12" y1="20" x2="12" y2="4"/>
        <line x1="6" y1="20" x2="6" y2="14"/>
    </svg>""",
    
    "alert": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ff4c4c" stroke-width="2">
        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
        <line x1="12" y1="9" x2="12" y2="13"/>
        <line x1="12" y1="17" x2="12.01" y2="17"/>
    </svg>""",
    
    "check": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#00ff88" stroke-width="2">
        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
        <polyline points="22 4 12 14.01 9 11.01"/>
    </svg>""",
    
    "upload": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#00ffff" stroke-width="2">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
        <polyline points="17 8 12 3 7 8"/>
        <line x1="12" y1="3" x2="12" y2="15"/>
    </svg>""",
    
    "download": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#00ffff" stroke-width="2">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
        <polyline points="7 10 12 15 17 10"/>
        <line x1="12" y1="15" x2="12" y2="3"/>
    </svg>""",
    
    "brain": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#00ffff" stroke-width="2">
        <path d="M12 2a4 4 0 0 0-4 4c0 1.1.9 2 2 2h4a2 2 0 0 0 2-2 4 4 0 0 0-4-4z"/>
        <path d="M12 22c-4.4 0-8-3.6-8-8 0-2.2.9-4.2 2.3-5.7"/>
        <path d="M12 22c4.4 0 8-3.6 8-8 0-2.2-.9-4.2-2.3-5.7"/>
        <path d="M12 8v6"/>
        <path d="M9 11h6"/>
    </svg>""",
    
    "activity": """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#00ff88" stroke-width="2">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
    </svg>"""
}

def get_icon(name):
    """Mengambil ikon SVG berdasarkan nama."""
    return ICONS.get(name, "")

def icon_with_text(icon_name, text, gap="10px"):
    """Membuat HTML untuk ikon dengan teks."""
    icon = get_icon(icon_name)
    return f"""<div style="display: flex; align-items: center; gap: {gap};">{icon}<span>{text}</span></div>"""
