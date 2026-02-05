import re

with open(r'c:\laragon\www\soc-streamlite\app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix hardcoded texts in render_soc_monitoring_page
replacements = [
    # SOC title and subtitle
    ('SOC Monitoring {mode_badge}', "{t('soc_title')} {mode_badge}"),
    ('Real-Time Network Traffic Analysis • AI-Powered Threat Detection', "{t('soc_subtitle')}"),
    
    # Auto refresh
    ('"🔄 Auto Refresh"', 'f"🔄 {t(\'auto_refresh\')}"'),
    ('"Interval"', 't(\'interval\')'),
    ('f"{x} detik"', 'f"{x} {t(\'seconds\')}"'),
    
    # Metrics
    ('"Total Connections"', 't(\'total_connections\')'),
    ('"Normal Traffic"', 't(\'normal_traffic\')'),
    ('"🚨 Threats Detected"', 'f"🚨 {t(\'threats_detected_count\')}"'),
    ('"Unique Sources"', 't(\'unique_sources\')'),
    ('"Data Transfer"', 't(\'data_transfer\')'),
    
    # Charts
    ('"📊 Protocol Distribution"', 'f"📊 {t(\'protocol_distribution\')}"'),
    ('"🎯 Threat Distribution"', 'f"🎯 {t(\'threat_distribution\')}"'),
    ("labels=['Normal', 'Attack']", "labels=[t('normal'), t('attack')]"),
    
    # Table
    ('"🔍 Live Connection Analysis"', 'f"🔍 {t(\'live_connection_analysis\')}"'),
    ('"Filter"', 't(\'filter_label\')'),
    ('["All", "Attacks Only", "Normal Only"]', '[t(\'filter_all\'), t(\'filter_attacks\'), t(\'filter_normal\')]'),
    
    # Alerts  
    ('"🚨 Active Threat Alerts"', 'f"🚨 {t(\'active_threat_alerts\')}"'),
    ('Potential {conn_type.upper()} Attack', "{t('potential_attack')} {conn_type.upper()}"),
    ('"No connections to display"', 't(\'no_connections\')'),
    
    # Sidebar - Operation Mode
    ('Mode Operasi', "{t('operation_mode')}"),
    ('"🎮 Enable Simulation Mode"', 'f"🎮 {t(\'enable_simulation\')}"'),
    ('"Centang untuk menggunakan data dummy (tanpa router fisik)"', 't(\'simulation_help\')'),
    ('"📡 Mode Simulasi Aktif"', 'f"📡 {t(\'simulation_active\')}"'),
    ('"🔌 Mode Real - Perlu Router MikroTik"', 'f"🔌 {t(\'real_mode_warning\')}"'),
    ('"**Konfigurasi Router:**"', 'f"**{t(\'router_config\')}:**"'),
    ('"IP Address"', 't(\'ip_address\')'),
    ('"Username"', 't(\'username\')'),
    ('"Password"', 't(\'password\')'),
    ('"API Port"', 't(\'api_port\')'),
    ('"🔗 Connect"', 'f"🔗 {t(\'connect_btn\')}"'),
    
    # Menu
    ('"🖥️ SOC Monitoring"', 'f"🖥️ {t(\'menu_soc\')}"'),
]

for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
        print(f'Replaced: {old[:40]}...')

with open(r'c:\laragon\www\soc-streamlite\app.py', 'w', encoding='utf-8', newline='') as f:
    f.write(content)

print('\\nAll translations applied!')
