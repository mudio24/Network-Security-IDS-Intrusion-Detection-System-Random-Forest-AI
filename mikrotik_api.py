# =============================================================================
# MIKROTIK API MODULE - HYBRID SIMULATION MODE
# CyberGuard AI: Enterprise Edition
# =============================================================================
# Supports both simulation mode (dummy data) and real MikroTik connection
# =============================================================================

import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

# Try to import routeros_api, but don't fail if not available
try:
    import routeros_api
    ROUTEROS_AVAILABLE = True
except ImportError:
    ROUTEROS_AVAILABLE = False


class MikroTikConnection:
    """
    MikroTik connection handler with Hybrid Simulation Mode.
    
    Supports:
    - Simulation Mode: Generate realistic dummy data for testing
    - Real Mode: Connect to actual MikroTik router via RouterOS API
    """
    
    def __init__(self, simulation_mode: bool = True):
        self.simulation_mode = simulation_mode
        self.connection = None
        self.api = None
        self.is_connected = False
        self.router_info = {}
        
        # Simulation parameters
        self._attack_ips = [
            "45.33.32.156", "185.220.101.35", "89.248.167.131",
            "141.98.10.60", "192.241.216.17", "185.156.73.54"
        ]
        self._normal_ips = [
            "192.168.1.10", "192.168.1.20", "192.168.1.30",
            "192.168.1.40", "192.168.1.50", "192.168.1.100",
            "192.168.1.101", "192.168.1.102", "192.168.1.103"
        ]
        self._protocols = ["tcp", "udp", "icmp"]
        self._services = {
            80: "http", 443: "https", 22: "ssh", 21: "ftp",
            25: "smtp", 53: "dns", 23: "telnet", 3389: "rdp",
            8080: "http-proxy", 3306: "mysql", 5432: "postgresql"
        }
    
    def connect(self, ip: str = None, username: str = None, 
                password: str = None, port: int = 8728) -> Tuple[bool, str]:
        """
        Connect to MikroTik router.
        
        In simulation mode, always returns success.
        In real mode, attempts actual connection via RouterOS API.
        """
        if self.simulation_mode:
            self.is_connected = True
            self.router_info = {
                "identity": "MikroTik-Simulator",
                "version": "7.x (Simulation)",
                "uptime": "5d 12:34:56",
                "cpu-load": f"{random.randint(5, 45)}%",
                "free-memory": f"{random.randint(128, 512)} MiB",
                "board-name": "RB4011iGS+ (Simulation)"
            }
            return True, "Connected to Simulation Mode"
        
        # Real mode - require routeros_api
        if not ROUTEROS_AVAILABLE:
            return False, "routeros-api library not installed. Run: pip install routeros-api"
        
        if not all([ip, username, password]):
            return False, "Missing required credentials (IP, Username, Password)"
        
        try:
            self.connection = routeros_api.RouterOsApiPool(
                ip,
                username=username,
                password=password,
                port=port,
                plaintext_login=True
            )
            self.api = self.connection.get_api()
            
            # Get router identity
            identity = self.api.get_resource('/system/identity')
            resource = self.api.get_resource('/system/resource')
            
            identity_data = identity.get()[0] if identity.get() else {}
            resource_data = resource.get()[0] if resource.get() else {}
            
            self.router_info = {
                "identity": identity_data.get('name', 'Unknown'),
                "version": resource_data.get('version', 'Unknown'),
                "uptime": resource_data.get('uptime', 'Unknown'),
                "cpu-load": resource_data.get('cpu-load', '0') + '%',
                "free-memory": resource_data.get('free-memory', 'Unknown'),
                "board-name": resource_data.get('board-name', 'Unknown')
            }
            
            self.is_connected = True
            return True, f"Connected to {self.router_info['identity']}"
            
        except Exception as e:
            self.is_connected = False
            return False, f"Connection failed: {str(e)}"
    
    def disconnect(self):
        """Disconnect from router."""
        if self.connection and not self.simulation_mode:
            try:
                self.connection.disconnect()
            except:
                pass
        self.is_connected = False
        self.connection = None
        self.api = None
    
    def get_system_info(self) -> Dict:
        """Get router system information."""
        if self.simulation_mode:
            # Update simulated CPU load dynamically
            self.router_info["cpu-load"] = f"{random.randint(5, 45)}%"
            return self.router_info
        
        if not self.is_connected or not self.api:
            return {}
        
        try:
            resource = self.api.get_resource('/system/resource')
            data = resource.get()[0] if resource.get() else {}
            return {
                "identity": self.router_info.get('identity', 'Unknown'),
                "version": data.get('version', 'Unknown'),
                "uptime": data.get('uptime', 'Unknown'),
                "cpu-load": data.get('cpu-load', '0') + '%',
                "free-memory": data.get('free-memory', 'Unknown'),
                "board-name": data.get('board-name', 'Unknown')
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_firewall_connections(self, simulation_mode: bool = None) -> List[Dict]:
        """
        Get active firewall/NAT connections.
        
        Args:
            simulation_mode: Override instance simulation mode if provided
            
        Returns:
            List of connection dictionaries matching MikroTik API format
        """
        use_simulation = simulation_mode if simulation_mode is not None else self.simulation_mode
        
        if use_simulation:
            return self._generate_mock_connections()
        
        if not self.is_connected or not self.api:
            return []
        
        try:
            # Get connection tracking table
            connections = self.api.get_resource('/ip/firewall/connection')
            return connections.get()
        except Exception as e:
            return [{"error": str(e)}]
    
    def _generate_mock_connections(self) -> List[Dict]:
        """
        Generate realistic mock connection data.
        
        Creates a mix of:
        - Normal traffic (80%)
        - Attack patterns (20%): High connection count, port scanning, etc.
        """
        connections = []
        num_connections = random.randint(50, 150)
        
        # Determine attack scenario
        attack_scenario = random.choice([
            "port_scan",      # Many connections to different ports
            "ddos",           # Many connections from same IP
            "brute_force",    # Many SSH/Telnet connections
            "normal_heavy"    # Heavy but normal traffic
        ])
        
        for i in range(num_connections):
            is_attack = random.random() < 0.20  # 20% chance of attack traffic
            
            if is_attack:
                conn = self._generate_attack_connection(attack_scenario)
            else:
                conn = self._generate_normal_connection()
            
            connections.append(conn)
        
        # Add specific attack patterns based on scenario
        if attack_scenario == "port_scan":
            attacker_ip = random.choice(self._attack_ips)
            target_ip = random.choice(self._normal_ips)
            for port in random.sample(range(1, 65535), random.randint(50, 100)):
                connections.append({
                    "src-address": f"{attacker_ip}:{random.randint(40000, 60000)}",
                    "dst-address": f"{target_ip}:{port}",
                    "protocol": "tcp",
                    "orig-bytes": str(random.randint(40, 100)),
                    "repl-bytes": "0",
                    "orig-packets": "1",
                    "repl-packets": "0",
                    "timeout": "1s",
                    "tcp-state": "syn-sent",
                    "connection-type": "port-scan"
                })
        
        elif attack_scenario == "ddos":
            target_ip = random.choice(self._normal_ips)
            for _ in range(random.randint(100, 200)):
                connections.append({
                    "src-address": f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}:{random.randint(1024, 65535)}",
                    "dst-address": f"{target_ip}:80",
                    "protocol": "tcp",
                    "orig-bytes": str(random.randint(1000, 50000)),
                    "repl-bytes": str(random.randint(0, 100)),
                    "orig-packets": str(random.randint(100, 1000)),
                    "repl-packets": str(random.randint(0, 10)),
                    "timeout": "30s",
                    "tcp-state": "established",
                    "connection-type": "ddos"
                })
        
        elif attack_scenario == "brute_force":
            attacker_ip = random.choice(self._attack_ips)
            target_ip = random.choice(self._normal_ips)
            for _ in range(random.randint(50, 150)):
                port = random.choice([22, 23, 3389])  # SSH, Telnet, RDP
                connections.append({
                    "src-address": f"{attacker_ip}:{random.randint(40000, 60000)}",
                    "dst-address": f"{target_ip}:{port}",
                    "protocol": "tcp",
                    "orig-bytes": str(random.randint(100, 500)),
                    "repl-bytes": str(random.randint(50, 200)),
                    "orig-packets": str(random.randint(5, 20)),
                    "repl-packets": str(random.randint(3, 15)),
                    "timeout": "5s",
                    "tcp-state": "established",
                    "connection-type": "brute-force"
                })
        
        return connections
    
    def _generate_normal_connection(self) -> Dict:
        """Generate a single normal connection."""
        src_ip = random.choice(self._normal_ips)
        dst_port = random.choice(list(self._services.keys()))
        dst_ip = f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
        
        return {
            "src-address": f"{src_ip}:{random.randint(1024, 65535)}",
            "dst-address": f"{dst_ip}:{dst_port}",
            "protocol": "tcp" if dst_port not in [53] else random.choice(["tcp", "udp"]),
            "orig-bytes": str(random.randint(100, 50000)),
            "repl-bytes": str(random.randint(100, 100000)),
            "orig-packets": str(random.randint(5, 100)),
            "repl-packets": str(random.randint(5, 200)),
            "timeout": f"{random.randint(30, 300)}s",
            "tcp-state": random.choice(["established", "time-wait", "close-wait"]),
            "connection-type": "normal"
        }
    
    def _generate_attack_connection(self, scenario: str) -> Dict:
        """Generate a single attack connection."""
        src_ip = random.choice(self._attack_ips)
        dst_ip = random.choice(self._normal_ips)
        
        if scenario == "port_scan":
            dst_port = random.randint(1, 65535)
            return {
                "src-address": f"{src_ip}:{random.randint(40000, 60000)}",
                "dst-address": f"{dst_ip}:{dst_port}",
                "protocol": "tcp",
                "orig-bytes": str(random.randint(40, 100)),
                "repl-bytes": "0",
                "orig-packets": "1",
                "repl-packets": "0",
                "timeout": "1s",
                "tcp-state": "syn-sent",
                "connection-type": "suspicious"
            }
        elif scenario == "ddos":
            return {
                "src-address": f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}:{random.randint(1024, 65535)}",
                "dst-address": f"{dst_ip}:80",
                "protocol": random.choice(["tcp", "udp"]),
                "orig-bytes": str(random.randint(5000, 100000)),
                "repl-bytes": str(random.randint(0, 500)),
                "orig-packets": str(random.randint(500, 5000)),
                "repl-packets": str(random.randint(0, 50)),
                "timeout": "60s",
                "tcp-state": "established",
                "connection-type": "flood"
            }
        else:  # brute_force
            port = random.choice([22, 23, 3389, 21])
            return {
                "src-address": f"{src_ip}:{random.randint(40000, 60000)}",
                "dst-address": f"{dst_ip}:{port}",
                "protocol": "tcp",
                "orig-bytes": str(random.randint(100, 1000)),
                "repl-bytes": str(random.randint(50, 500)),
                "orig-packets": str(random.randint(10, 50)),
                "repl-packets": str(random.randint(5, 30)),
                "timeout": "10s",
                "tcp-state": "established",
                "connection-type": "brute-force"
            }
    
    def get_interface_traffic(self) -> List[Dict]:
        """Get interface traffic statistics."""
        if self.simulation_mode:
            return self._generate_mock_interface_traffic()
        
        if not self.is_connected or not self.api:
            return []
        
        try:
            interfaces = self.api.get_resource('/interface')
            return interfaces.get()
        except Exception as e:
            return [{"error": str(e)}]
    
    def _generate_mock_interface_traffic(self) -> List[Dict]:
        """Generate mock interface traffic data."""
        interfaces = ["ether1-WAN", "ether2-LAN", "wlan1", "bridge1"]
        traffic = []
        
        for iface in interfaces:
            traffic.append({
                "name": iface,
                "type": "ether" if "ether" in iface else ("wlan" if "wlan" in iface else "bridge"),
                "rx-byte": str(random.randint(1000000000, 50000000000)),
                "tx-byte": str(random.randint(500000000, 20000000000)),
                "rx-packet": str(random.randint(1000000, 50000000)),
                "tx-packet": str(random.randint(500000, 20000000)),
                "running": "true",
                "disabled": "false"
            })
        
        return traffic


def parse_connection_for_ai(connection: Dict) -> Dict:
    """
    Parse MikroTik connection data to AI model input format.
    
    Converts MikroTik API format to the feature format expected by the AI model:
    - protocol_type: tcp/udp/icmp
    - src_bytes: bytes from source
    - count: estimated connection count (based on packets)
    
    Returns dictionary ready for AI prediction.
    """
    # Extract source bytes
    src_bytes = int(connection.get("orig-bytes", 0))
    
    # Extract protocol
    protocol = connection.get("protocol", "tcp").lower()
    
    # Estimate connection count from packets
    orig_packets = int(connection.get("orig-packets", 1))
    repl_packets = int(connection.get("repl-packets", 0))
    count = orig_packets + repl_packets
    
    # Extract destination bytes
    dst_bytes = int(connection.get("repl-bytes", 0))
    
    # Extract service from destination port
    dst_address = connection.get("dst-address", "0.0.0.0:80")
    try:
        dst_port = int(dst_address.split(":")[-1])
    except:
        dst_port = 80
    
    # Map port to service name
    port_to_service = {
        80: "http", 443: "http", 22: "ssh", 21: "ftp",
        25: "smtp", 53: "domain", 23: "telnet", 110: "pop3",
        3389: "X11", 8080: "http"
    }
    service = port_to_service.get(dst_port, "private")
    
    # Determine flag based on TCP state
    tcp_state = connection.get("tcp-state", "established")
    state_to_flag = {
        "established": "SF",
        "syn-sent": "S0",
        "syn-received": "S1",
        "time-wait": "SF",
        "close-wait": "SF",
        "close": "SF",
        "fin-wait-1": "SF",
        "fin-wait-2": "SF"
    }
    flag = state_to_flag.get(tcp_state, "SF")
    
    return {
        "duration": 0,  # Not available from connection tracking
        "protocol_type": protocol,
        "service": service,
        "flag": flag,
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "count": min(count, 500),  # Cap at 500
        "srv_count": min(count, 500),
        
        # Additional metadata for display
        "src_address": connection.get("src-address", "Unknown"),
        "dst_address": connection.get("dst-address", "Unknown"),
        "connection_type": connection.get("connection-type", "unknown"),
        "tcp_state": tcp_state
    }


def get_connection_stats(connections: List[Dict]) -> Dict:
    """
    Calculate statistics from connections list.
    
    Returns aggregated stats for dashboard display.
    """
    if not connections:
        return {
            "total": 0,
            "tcp": 0,
            "udp": 0,
            "icmp": 0,
            "total_bytes_in": 0,
            "total_bytes_out": 0,
            "unique_sources": 0,
            "unique_destinations": 0
        }
    
    stats = {
        "total": len(connections),
        "tcp": 0,
        "udp": 0,
        "icmp": 0,
        "total_bytes_in": 0,
        "total_bytes_out": 0,
        "unique_sources": set(),
        "unique_destinations": set()
    }
    
    for conn in connections:
        protocol = conn.get("protocol", "tcp").lower()
        if protocol == "tcp":
            stats["tcp"] += 1
        elif protocol == "udp":
            stats["udp"] += 1
        elif protocol == "icmp":
            stats["icmp"] += 1
        
        stats["total_bytes_in"] += int(conn.get("orig-bytes", 0))
        stats["total_bytes_out"] += int(conn.get("repl-bytes", 0))
        
        src = conn.get("src-address", "").split(":")[0]
        dst = conn.get("dst-address", "").split(":")[0]
        if src:
            stats["unique_sources"].add(src)
        if dst:
            stats["unique_destinations"].add(dst)
    
    stats["unique_sources"] = len(stats["unique_sources"])
    stats["unique_destinations"] = len(stats["unique_destinations"])
    
    return stats
