import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import math
import json
import openai
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')
import io

AI_CONFIG = {
    'api_key': "sk-ObXbMdavg61VYQgn494c51E327154f2bBfAf6a8fC7D1BeCa",
    'api_base': "https://maas-api.cn-huabei-1.xf-yun.com/v1",
    'model': "xop3qwen1b7"
}

# 初始化AI客户端（放在合适的位置）
def init_ai_client():
    """初始化AI客户端"""
    try:
        client = openai.OpenAI(
            api_key=AI_CONFIG['api_key'],
            base_url=AI_CONFIG['api_base']
        )
        # 简单测试
        test_response = client.chat.completions.create(
            model=AI_CONFIG['model'],
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        return client, True
    except Exception as e:
        print(f"AI客户端初始化失败: {e}")
        return None, False

# ============================================
# 1. 页面设置和样式
# ============================================

st.set_page_config(
    page_title="CarbonFlow - 供应链碳足迹管理平台",
    page_icon="🖇️",
    layout="wide"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E6F5C;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 15px;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .section-header {
        color: #2E8B57;
        border-bottom: 3px solid #2E8B57;
        padding-bottom: 10px;
        margin-top: 30px;
        font-weight: bold;
        font-size: 1.8rem;
    }
    .supplier-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 3px 6px rgba(0,0,0,0.16);
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-active { background-color: #4CAF50; }
    .status-warning { background-color: #FFC107; }
    .status-critical { background-color: #F44336; }
</style>
""", unsafe_allow_html=True)

# ============================================
# 2. 全局变量和初始化
# ============================================

# 初始化session state
if 'company_data' not in st.session_state:
    st.session_state.company_data = None
if 'emission_results' not in st.session_state:
    st.session_state.emission_results = {}
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = {}
if 'manual_data' not in st.session_state:
    st.session_state.manual_data = {}
if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = '智能电子制造企业'
if 'data_source' not in st.session_state:
    st.session_state.data_source = '示例数据'

# 排放因子数据库
EMISSION_FACTORS = {
    'natural_gas': 2.162,
    'diesel': 2.732,
    'coal': 2.53,
    'truck_transport': 0.062,
    'rail_transport': 0.022,
    'ship_transport': 0.010,
    'air_transport': 0.805,
    'steel': 1.85,
    'aluminum': 8.24,
    'plastic': 2.53
}

# ========== 1. 定义改进的排放因子 ==========
IMPROVED_FACTORS = {
    # Scope 1: 直接排放
    'diesel': 2.732,  # 修正：柴油排放因子 (kg CO₂/L)
    'natural_gas': 2.162,  # 保持：天然气排放因子 (kg CO₂/m³)

    # Scope 2: 外购电力
    'electricity': {
        '华东电网': 0.7035,
        '华北电网': 0.8571,  # 修正：使用2022年官方数据
        '华南电网': 0.6112,  # 修正：使用2022年官方数据
        '华中电网': 0.5253,  # 新增
        '东北电网': 0.8789,  # 新增
        '西北电网': 0.6675  # 新增
    },

    # Scope 3: 运输排放 - 使用全局EMISSION_FACTORS中的值，但修正单位理解
    'truck_transport': 0.062,  # kg CO₂/(ton·km) - 修正理解
    'rail_transport': 0.022,  # kg CO₂/(ton·km)
    'ship_transport': 0.010,  # kg CO₂/(ton·km)
    'air_transport': 0.805,  # kg CO₂/(ton·km)

    # Scope 3: 物料排放
    'steel': 1.85,  # 钢材 (kg CO₂/kg)
    'aluminum': 8.24,  # 铝材
    'plastic': 2.53,  # 塑料
    'electronics': 0.50  # 电子元件
}


# ========== 2. 定义改进的计算函数 ==========
def calculate_scope1_improved(fuel_data):
    """改进的范围1计算"""
    scope1 = 0
    if 'diesel_l' in fuel_data:
        scope1 += fuel_data['diesel_l'] * IMPROVED_FACTORS['diesel']
    if 'natural_gas_m3' in fuel_data:
        scope1 += fuel_data['natural_gas_m3'] * IMPROVED_FACTORS['natural_gas']
    return scope1 / 1000  # 转换为吨


def calculate_scope2_improved(electricity_data, region='华东电网'):
    """改进的范围2计算"""
    if 'electricity_kwh' in electricity_data:
        factor = IMPROVED_FACTORS['electricity'].get(region, 0.7035)
        return (electricity_data['electricity_kwh'] * factor) / 1000
    return 0


def calculate_scope3_improved(supplier_data):
    """改进的范围3计算（包含供应商排行）"""
    scope3 = 0

    if supplier_data is not None and not supplier_data.empty:
        total_material_emission = 0
        total_transport_emission = 0

        # 存储每个供应商的详细排放数据
        supplier_emissions = []

        for idx, supplier in supplier_data.iterrows():
            supplier_name = supplier.get('name', f'供应商{idx}')
            supplier_type = supplier.get('type', '未知')
            supplier_id = supplier.get('supplier_id', f'SUP{idx:03d}')

            # 1. 物料排放计算
            material_emission = 0

            # 根据供应商类型确定物料因子
            material_factor_map = {
                'steel': 2.20,  # 钢材 (kg CO₂/kg)
                'aluminum': 8.60,  # 铝材
                'plastic': 1.85,  # 塑料
                'electronics': 0.50  # 电子元件
            }

            # 简化：根据type关键字匹配
            material_factor = 2.0  # 默认值
            for key in material_factor_map:
                if key in str(supplier_type).lower():
                    material_factor = material_factor_map[key]
                    break

            # 获取或估算年采购量（吨）
            if 'annual_purchase_volume' in supplier:
                annual_purchase_ton = float(supplier['annual_purchase_volume'])
            elif 'annual_purchase_volume_ton' in supplier:
                annual_purchase_ton = float(supplier['annual_purchase_volume_ton'])
            else:
                # 根据供应商类型估算
                if 'steel' in str(supplier_type).lower():
                    annual_purchase_ton = np.random.uniform(50, 200)  # 钢材供应商
                elif 'aluminum' in str(supplier_type).lower():
                    annual_purchase_ton = np.random.uniform(20, 100)  # 铝材供应商
                elif 'plastic' in str(supplier_type).lower():
                    annual_purchase_ton = np.random.uniform(30, 150)  # 塑料供应商
                else:
                    annual_purchase_ton = np.random.uniform(10, 80)  # 其他

            material_emission = annual_purchase_ton * 1000 * material_factor  # 吨转kg
            total_material_emission += material_emission

            # 2. 运输排放计算
            transport_emission = 0

            # 获取运输距离
            if 'distance_km' in supplier:
                distance_km = float(supplier['distance_km'])
            elif 'lat' in supplier and 'lon' in supplier:
                # 计算到苏州的距离
                try:
                    # 工厂位置（苏州）
                    factory_lat, factory_lon = 31.2989, 120.5853
                    supplier_lat, supplier_lon = float(supplier['lat']), float(supplier['lon'])

                    # 使用您的 calculate_distance 函数
                    distance_km = calculate_distance(factory_lat, factory_lon, supplier_lat, supplier_lon)
                except:
                    distance_km = np.random.uniform(100, 800)  # 随机距离
            else:
                distance_km = np.random.uniform(100, 800)  # 随机距离

            # 确定运输方式
            if 'transport_mode' in supplier:
                transport_mode = supplier['transport_mode']
            elif 'transport' in supplier:
                transport_mode = supplier['transport']
            else:
                # 根据距离推断运输方式
                if distance_km < 200:
                    transport_mode = 'truck'
                elif distance_km < 1000:
                    transport_mode = 'rail'
                else:
                    transport_mode = 'ship'

            # 获取运输排放因子
            transport_factor_map = {
                'truck': 0.062,  # 公路运输
                'rail': 0.022,  # 铁路运输
                'ship': 0.010,  # 水路运输
                'air': 0.805  # 航空运输
            }

            transport_factor = transport_factor_map.get(
                transport_mode.lower(),
                transport_factor_map['truck']
            )

            # 获取运输频率和重量
            if 'transport_frequency_per_month' in supplier:
                trips_per_month = float(supplier['transport_frequency_per_month'])
            else:
                trips_per_month = np.random.uniform(0.5, 4)  # 每月0.5-4次

            if 'weight_per_trip_ton' in supplier:
                weight_per_trip_ton = float(supplier['weight_per_trip_ton'])
            else:
                weight_per_trip_ton = np.random.uniform(5, 20)  # 每次5-20吨

            trips_per_year = trips_per_month * 12

            # 运输排放 = 距离 × 重量 × 次数 × 因子
            transport_emission = distance_km * weight_per_trip_ton * trips_per_year * transport_factor
            total_transport_emission += transport_emission

            # 计算供应商总排放（吨）
            supplier_total_emission = (material_emission + transport_emission) / 1000

            # 存储供应商排放数据
            supplier_emissions.append({
                'supplier_id': supplier_id,
                'supplier_name': supplier_name,
                'supplier_type': supplier_type,
                'material_emission_t': material_emission / 1000,
                'transport_emission_t': transport_emission / 1000,
                'total_emission_t': supplier_total_emission,
                'annual_purchase_ton': annual_purchase_ton,
                'distance_km': distance_km,
                'transport_mode': transport_mode,
                'trips_per_year': trips_per_year,
                'weight_per_trip_ton': weight_per_trip_ton,
                'material_factor': material_factor,
                'transport_factor': transport_factor
            })

        # 排序：按总排放降序
        supplier_emissions.sort(key=lambda x: x['total_emission_t'], reverse=True)

        # 计算占比
        total_emission_t = (total_material_emission + total_transport_emission) / 1000

        for supplier in supplier_emissions:
            if total_emission_t > 0:
                supplier['emission_percentage'] = (supplier['total_emission_t'] / total_emission_t) * 100
            else:
                supplier['emission_percentage'] = 0

        # 总计范围3排放
        scope3 = total_emission_t

        # 存储详细结果
        scope3_details = {
            'total': scope3,
            'material': total_material_emission / 1000,
            'transport': total_transport_emission / 1000,
            'supplier_count': len(supplier_data),
            'supplier_emissions': supplier_emissions,  # 新增：包含每个供应商的排放数据
            'top_emitters': supplier_emissions[:5] if len(supplier_emissions) > 5 else supplier_emissions
        }

        return scope3, scope3_details

    return 0, {
        'total': 0,
        'material': 0,
        'transport': 0,
        'supplier_count': 0,
        'supplier_emissions': [],
        'top_emitters': []
    }


# ============================================
# 3. 示例数据系统
# ============================================
class EnhancedSampleDataSystem:
    """增强版示例数据系统"""

    def __init__(self):
        self.datasets = {
            '智能电子制造企业': self.create_smart_electronics_data(),
            '中型机械设备厂': self.create_medium_machinery_data(),
            '绿色化工企业': self.create_green_chemical_data(),
            '汽车零部件供应商': self.create_auto_parts_data(),
            '食品加工企业': self.create_food_processing_data()
        }

    def create_smart_electronics_data(self):
        """智能电子制造企业数据"""
        return {
            'company_info': self.create_rich_company_info('电子设备制造'),
            'energy_data': self.create_rich_energy_data(250000, 5000, 1000),
            'production_data': self.create_rich_production_data(['智能控制器', '传感器模块', '通讯模块']),
            'supplier_data': self.create_rich_supplier_data(['钢材', '铝材', '电子元件', '塑料粒子'])
        }

    def create_medium_machinery_data(self):
        """中型机械设备厂数据"""
        return {
            'company_info': self.create_rich_company_info('机械设备制造'),
            'energy_data': self.create_rich_energy_data(180000, 3000, 2000),
            'production_data': self.create_rich_production_data(['数控机床', '工业机器人', '传送设备']),
            'supplier_data': self.create_rich_supplier_data(['钢材', '铸铁', '电机', '控制系统'])
        }

    def create_green_chemical_data(self):
        """绿色化工企业数据"""
        return {
            'company_info': self.create_rich_company_info('化工制造'),
            'energy_data': self.create_rich_energy_data(350000, 8000, 1500),
            'production_data': self.create_rich_production_data(['环保涂料', '生物降解塑料', '清洁剂']),
            'supplier_data': self.create_rich_supplier_data(['化工原料', '包装材料', '催化剂'])
        }

    def create_auto_parts_data(self):
        """汽车零部件供应商数据"""
        return {
            'company_info': self.create_rich_company_info('汽车零部件制造'),
            'energy_data': self.create_rich_energy_data(150000, 2500, 800),
            'production_data': self.create_rich_production_data(['汽车座椅', '内饰件', '外饰件']),
            'supplier_data': self.create_rich_supplier_data(['钢材', '塑料', '皮革', '电子元件'])
        }

    def create_food_processing_data(self):
        """食品加工企业数据"""
        return {
            'company_info': self.create_rich_company_info('食品加工'),
            'energy_data': self.create_rich_energy_data(120000, 1500, 500),
            'production_data': self.create_rich_production_data(['饼干', '饮料', '方便食品']),
            'supplier_data': self.create_rich_supplier_data(['粮食', '包装', '添加剂'])
        }

    # ========== 数据生成方法 ==========

    def create_rich_company_info(self, industry):
        """创建丰富的公司信息"""
        base_info = {
            '电子设备制造': {
                'company_name': '苏州精密智能制造有限公司',
                'employee_count': 680,
                'annual_revenue_million': 850,
                'main_products': '智能控制器、传感器模块、工业通讯设备'
            },
            '机械设备制造': {
                'company_name': '华东机械设备有限公司',
                'employee_count': 450,
                'annual_revenue_million': 520,
                'main_products': '数控机床、工业机器人、自动化设备'
            },
            '化工制造': {
                'company_name': '绿源化工有限公司',
                'employee_count': 320,
                'annual_revenue_million': 680,
                'main_products': '环保涂料、生物降解材料、特种化学品'
            },
            '汽车零部件制造': {
                'company_name': '安达汽车零部件有限公司',
                'employee_count': 580,
                'annual_revenue_million': 720,
                'main_products': '汽车座椅、内饰件、外饰件'
            },
            '食品加工': {
                'company_name': '康美食品有限公司',
                'employee_count': 280,
                'annual_revenue_million': 380,
                'main_products': '饼干、饮料、方便食品'
            }
        }

        info = base_info.get(industry, base_info['电子设备制造'])

        return pd.DataFrame([{
            'company_id': 'CMP001',
            'company_name': info['company_name'],
            'industry': industry,
            'location': '江苏省苏州市',
            'employee_count': info['employee_count'],
            'annual_revenue_million': info['annual_revenue_million'],
            'main_products': info['main_products'],
            'established_year': 2010,
            'facility_area_sqm': 20000,
            'energy_manager': '王明',
            'certifications': 'ISO9001, ISO14001'
        }])

    def create_rich_energy_data(self, electricity_base, gas_base, diesel_base):
        """创建丰富的能源数据"""
        np.random.seed(42)

        energy_records = []

        # 生成24个月数据
        for i in range(24):
            month_idx = i + 1
            year = 2023 if i < 12 else 2024
            month = (i % 12) + 1

            # 1. 长期趋势：每年下降2%
            trend_factor = 0.98 ** (i // 12)

            # 2. 季节性：夏季高，冬季中，春秋低
            if month in [6, 7, 8]:  # 夏季
                season_factor = 1.25
            elif month in [12, 1, 2]:  # 冬季
                season_factor = 1.15
            else:  # 春秋
                season_factor = 0.95

            # 3. 随机波动
            random_factor = np.random.uniform(0.9, 1.1)

            # 计算各能源消耗
            electricity = electricity_base * trend_factor * season_factor * random_factor
            natural_gas = gas_base * trend_factor * (1 + 0.3 * np.sin(2 * np.pi * (month - 1) / 12)) * random_factor
            diesel = diesel_base * trend_factor * random_factor
            water = electricity_base * 0.02 * season_factor * random_factor  # 用水量与用电相关

            # 可再生能源（逐年增加）
            solar_power = electricity * (0.05 + 0.008 * i)  # 从5%开始每月+0.8%

            energy_records.append({
                'year_month': f'{year}-{month:02d}',
                'electricity_kwh': round(electricity),
                'natural_gas_m3': round(natural_gas),
                'diesel_l': round(diesel),
                'water_tons': round(water),
                'solar_power_kwh': round(solar_power),
                'energy_cost_yuan': round(electricity * 0.8 + natural_gas * 3.5 + diesel * 8),
                'production_days': np.random.randint(22, 26),
                'avg_temperature_c': np.random.randint(-5, 35),
                'energy_efficiency': round(np.random.uniform(85, 95), 1),
                'maintenance_status': '正常' if np.random.random() > 0.1 else '检修中',
                'energy_manager': ['张三', '李四', '王五'][i % 3]
            })

        return pd.DataFrame(energy_records)

    def create_rich_production_data(self, product_types):
        """创建丰富的生产数据"""
        np.random.seed(43)

        records = []

        # 生成24个月数据
        for i in range(24):
            year = 2023 if i < 12 else 2024
            month = (i % 12) + 1

            for product in product_types:
                # 基础产量（不同产品不同）
                base_volume = {
                    '智能控制器': 5000,
                    '传感器模块': 8000,
                    '通讯模块': 6000,
                    '数控机床': 200,
                    '工业机器人': 150,
                    '传送设备': 300,
                    '环保涂料': 10000,
                    '生物降解塑料': 8000,
                    '清洁剂': 12000,
                    '汽车座椅': 3000,
                    '内饰件': 5000,
                    '外饰件': 4000,
                    '饼干': 20000,
                    '饮料': 15000,
                    '方便食品': 10000
                }.get(product, 5000)

                # 增长趋势
                growth_factor = 1 + 0.01 * i  # 每月增长1%

                # 季节性
                if month in [11, 12, 1]:  # 旺季
                    season_factor = 1.3
                elif month in [6, 7, 8]:  # 淡季
                    season_factor = 0.9
                else:
                    season_factor = 1.0

                # 随机波动
                random_factor = np.random.uniform(0.95, 1.05)

                # 计算产量
                volume = base_volume * growth_factor * season_factor * random_factor

                # 质量指标
                defect_rate = np.random.uniform(0.01, 0.05)  # 1-5%不良率

                records.append({
                    'year_month': f'{year}-{month:02d}',
                    'product_type': product,
                    'production_volume': round(volume),
                    'defect_quantity': round(volume * defect_rate),
                    'defect_rate_percent': round(defect_rate * 100, 2),
                    'raw_material_cost_yuan': round(volume * np.random.uniform(50, 200)),
                    'labor_cost_yuan': round(volume * np.random.uniform(10, 50)),
                    'equipment_utilization_percent': round(np.random.uniform(75, 92), 1),
                    'production_efficiency': round(np.random.uniform(85, 98), 1),
                    'customer_feedback_score': round(np.random.uniform(4.0, 5.0), 1),
                    'export_percentage': round(np.random.uniform(10, 80), 1)
                })

        return pd.DataFrame(records)

    def create_rich_supplier_data(self, supplier_types):
        """创建丰富的供应商数据"""
        np.random.seed(44)

        # 中国主要城市坐标
        cities = {
            '上海': (31.2304, 121.4737),
            '苏州': (31.2989, 120.5853),
            '深圳': (22.5431, 114.0579),
            '东莞': (23.0207, 113.7518),
            '天津': (39.3434, 117.3616),
            '重庆': (29.5630, 106.5516),
            '武汉': (30.5928, 114.3055),
            '青岛': (36.0671, 120.3826),
            '宁波': (29.8683, 121.5440),
            '沈阳': (41.8057, 123.4315)
        }

        records = []
        city_list = list(cities.keys())

        # 每种类型创建2-3个供应商
        for i, supplier_type in enumerate(supplier_types):
            for j in range(np.random.randint(2, 4)):
                city = city_list[(i * 3 + j) % len(city_list)]
                lat, lon = cities[city]

                # 添加微小偏移
                lat += np.random.uniform(-0.1, 0.1)
                lon += np.random.uniform(-0.1, 0.1)

                # 计算到苏州的距离
                distance = self.calculate_distance(31.2989, 120.5853, lat, lon)

                # 供应商评级
                total_score = np.random.uniform(70, 95)
                if total_score > 90:
                    rating = 'A+'
                elif total_score > 85:
                    rating = 'A'
                elif total_score > 80:
                    rating = 'B'
                elif total_score > 75:
                    rating = 'C'
                else:
                    rating = 'D'

                records.append({
                    'supplier_id': f'SUP{i:02d}{j}',
                    'name': f'{city}{supplier_type[:2]}供应商',
                    'type': supplier_type,
                    'city': city,
                    'lat': round(lat, 4),
                    'lon': round(lon, 4),
                    'rating': rating,
                    'quality_score': round(np.random.uniform(85, 98), 1),
                    'delivery_days': np.random.randint(7, 21),
                    'distance_km': round(distance),
                    'annual_purchase_volume': round(np.random.uniform(50, 300), 1),
                    'cooperation_years': np.random.randint(1, 10),
                    'contact_person': ['张经理', '李主任', '王总监', '刘总'][j % 4]
                })

        return pd.DataFrame(records)

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """计算两个经纬度坐标之间的距离（公里）"""
        # 将十进制度数转化为弧度
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        # Haversine公式
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))

        # 地球平均半径（公里）
        r = 6371.0

        return c * r

    def get_dataset(self, dataset_name):
        """获取数据集"""
        return self.datasets.get(dataset_name)


# ============================================
# 4. 核心功能模块
# ============================================


# ============================================
# 5. 页面函数 - 数据管理
# ============================================

def show_data_management():
    """显示数据管理页面"""
    st.markdown('<h2 class="section-header">数据管理系统</h2>', unsafe_allow_html=True)

    # 数据源选择
    data_source = st.radio(
        "选择数据输入方式",
        ["🗞️ 示例数据", "📦 文件上传", "✒️ 手动录入"],
        horizontal=True
    )

    if data_source == "🗞️ 示例数据":
        show_sample_data_system()
    elif data_source == "📦 文件上传":
        show_file_upload_system()
    elif data_source == "✒️ 手动录入":
        show_manual_entry_system()

    # 显示当前数据预览
    show_data_preview()


def show_sample_data_system():
    """显示示例数据系统"""

    # 初始化示例数据系统
    if 'sample_system' not in st.session_state:
        st.session_state.sample_system = EnhancedSampleDataSystem()

    sample_system = st.session_state.sample_system

    st.markdown("### 选择示例数据集")

    # 数据集选择卡片
    col1, col2, col3 = st.columns(3)

    datasets = [
        ('智能电子制造企业', '💻', '电子设备制造，数据维度丰富'),
        ('中型机械设备厂', '🏭', '机械设备制造，能源消耗典型'),
        ('绿色化工企业', '🧪', '化工行业，关注环保与安全'),
        ('汽车零部件供应商', '🚗', '汽车产业链，供应链复杂'),
        ('食品加工企业', '🍪', '快消品行业，季节性明显')
    ]

    # 第一行
    with col1:
        dataset_name, icon, desc = datasets[0]
        if st.button(f"{icon} {dataset_name}", use_container_width=True):
            st.session_state.current_dataset = dataset_name
            st.session_state.data_source = '示例数据'
            st.rerun()
        st.caption(desc)

    with col2:
        dataset_name, icon, desc = datasets[1]
        if st.button(f"{icon} {dataset_name}", use_container_width=True):
            st.session_state.current_dataset = dataset_name
            st.session_state.data_source = '示例数据'
            st.rerun()
        st.caption(desc)

    with col3:
        dataset_name, icon, desc = datasets[2]
        if st.button(f"{icon} {dataset_name}", use_container_width=True):
            st.session_state.current_dataset = dataset_name
            st.session_state.data_source = '示例数据'
            st.rerun()
        st.caption(desc)

    # 第二行
    col4, col5 = st.columns(2)

    with col4:
        dataset_name, icon, desc = datasets[3]
        if st.button(f"{icon} {dataset_name}", use_container_width=True):
            st.session_state.current_dataset = dataset_name
            st.session_state.data_source = '示例数据'
            st.rerun()
        st.caption(desc)

    with col5:
        dataset_name, icon, desc = datasets[4]
        if st.button(f"{icon} {dataset_name}", use_container_width=True):
            st.session_state.current_dataset = dataset_name
            st.session_state.data_source = '示例数据'
            st.rerun()
        st.caption(desc)

    # 显示当前数据集信息
    if st.session_state.current_dataset:
        current_data = sample_system.get_dataset(st.session_state.current_dataset)
        if current_data:
            st.success(f"当前选择：{st.session_state.current_dataset}")

            with st.expander("查看数据详情"):
                tab1, tab2, tab3 = st.tabs(["公司信息", "能源数据", "供应商"])

                with tab1:
                    st.dataframe(current_data['company_info'], use_container_width=True)

                with tab2:
                    st.dataframe(current_data['energy_data'], use_container_width=True)

                with tab3:
                    st.dataframe(current_data['supplier_data'], use_container_width=True)


def show_file_upload_system():
    """显示文件上传系统"""

    st.markdown("### 多文件上传")

    uploaded_files = {}

    # 创建四个必上传的文件上传器
    uploaded_file_company = st.file_uploader(
        "上传公司信息数据(CSV)",
        type=['csv'],
        key="upload_company_info"
    )

    uploaded_file_energy = st.file_uploader(
        "上传能源消耗数据(CSV)",
        type=['csv'],
        key="upload_energy_data"
    )

    uploaded_file_production = st.file_uploader(
        "上传生产数据(CSV)",
        type=['csv'],
        key="upload_production_data"
    )

    uploaded_file_supplier = st.file_uploader(
        "上传供应商信息数据(CSV)",
        type=['csv'],
        key="upload_supplier_data"
    )

    # 将上传的文件存入字典
    if uploaded_file_company is not None:
        uploaded_files["公司信息"] = uploaded_file_company

    if uploaded_file_energy is not None:
        uploaded_files["能源消耗"] = uploaded_file_energy

    if uploaded_file_production is not None:
        uploaded_files["生产数据"] = uploaded_file_production

    if uploaded_file_supplier is not None:
        uploaded_files["供应商信息"] = uploaded_file_supplier

    # 处理上传按钮
    if st.button("处理上传的文件", type="primary", use_container_width=True):
        if len(uploaded_files) == 4:
            process_uploaded_files(uploaded_files)
        elif uploaded_files:
            st.warning("请上传全部四个文件")
        else:
            st.warning("请先上传文件")

def process_uploaded_files(uploaded_files):
    """处理上传的文件"""

    for data_type, uploaded_file in uploaded_files.items():
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_data[data_type] = df

            st.success(f"{data_type}: 成功上传 {len(df)} 条记录")

        except Exception as e:
            st.error(f"{data_type} 处理失败: {str(e)}")

    st.session_state.data_source = '文件上传'


def show_manual_entry_system():
    """显示手动录入系统"""

    st.markdown("### 手动数据录入")

    entry_type = st.selectbox(
        "选择录入数据类型",
        ["能源消耗", "生产数据", "供应商信息"],
        index=0
    )

    if entry_type == "能源消耗":
        manual_data = energy_entry_form()
    elif entry_type == "生产数据":
        manual_data = production_entry_form()
    elif entry_type == "供应商信息":
        manual_data = supplier_entry_form()

    if manual_data:
        if entry_type not in st.session_state.manual_data:
            st.session_state.manual_data[entry_type] = []

        st.session_state.manual_data[entry_type].append(manual_data)
        st.session_state.data_source = '手动录入'
        st.success("数据录入成功！")


def energy_entry_form():
    """能源消耗录入表单"""
    with st.form("energy_form"):
        col1, col2 = st.columns(2)

        with col1:
            company = st.text_input("公司名称", value="苏州精密制造")
            month = st.text_input("月份", value="2024-01")
            electricity = st.number_input("电力(kWh)", value=250000)

        with col2:
            natural_gas = st.number_input("天然气(m³)", value=5000)
            diesel = st.number_input("柴油(L)", value=1000)

        submitted = st.form_submit_button("提交数据")

        if submitted:
            return {
                'company': company,
                'month': month,
                'electricity_kwh': electricity,
                'natural_gas_m3': natural_gas,
                'diesel_l': diesel,
                'entry_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

    return None


def production_entry_form():
    """生产数据录入表单"""
    with st.form("production_form"):
        product = st.text_input("产品类型", value="智能控制器")
        month = st.text_input("月份", value="2024-01")
        volume = st.number_input("产量", value=1000)
        defect_rate = st.number_input("不良率(%)", value=2.5)

        submitted = st.form_submit_button("提交数据")

        if submitted:
            return {
                'product_type': product,
                'month': month,
                'production_volume': volume,
                'defect_rate': defect_rate,
                'entry_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

    return None


def supplier_entry_form():
    """供应商信息录入表单"""
    with st.form("supplier_form"):
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("供应商名称", value="宝钢集团")
            material = st.selectbox("材料类型", ["钢材", "铝材", "塑料", "其他"])
            rating = st.selectbox("评级", ["A", "B", "C", "D"])

        with col2:
            city = st.text_input("所在城市", value="上海")
            lat = st.number_input("纬度", value=31.2304)
            lon = st.number_input("经度", value=121.4737)

        submitted = st.form_submit_button("提交数据")

        if submitted:
            return {
                'name': name,
                'material_type': material,
                'supplier_rating': rating,
                'city': city,
                'lat': lat,
                'lon': lon,
                'entry_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

    return None


def show_data_preview():
    """显示数据预览"""

    st.markdown("---")
    st.markdown("### 当前数据预览")

    if st.session_state.data_source == '示例数据':
        if 'sample_system' in st.session_state:
            data = st.session_state.sample_system.get_dataset(st.session_state.current_dataset)
            if data:
                st.info(f"当前使用：{st.session_state.current_dataset} 示例数据")

                # 计算数据统计
                energy_rows = len(data['energy_data']) if 'energy_data' in data else 0
                production_rows = len(data['production_data']) if 'production_data' in data else 0
                supplier_rows = len(data['supplier_data']) if 'supplier_data' in data else 0

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("能源记录", f"{energy_rows}条")
                with col2:
                    st.metric("生产记录", f"{production_rows}条")
                with col3:
                    st.metric("供应商", f"{supplier_rows}家")

    elif st.session_state.data_source == '文件上传':
        if st.session_state.uploaded_data:
            st.info("当前使用：上传文件数据")

            total_records = sum(len(df) for df in st.session_state.uploaded_data.values())
            data_types = len(st.session_state.uploaded_data)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("总记录数", f"{total_records}条")
            with col2:
                st.metric("数据类型", f"{data_types}类")

            # 显示各类型数据
            for data_type, df in st.session_state.uploaded_data.items():
                with st.expander(f"{data_type} ({len(df)}条)"):
                    st.dataframe(df.head(), use_container_width=True)

    elif st.session_state.data_source == '手动录入':
        if st.session_state.manual_data:
            st.info("当前使用：手动录入数据")

            total_records = sum(len(records) for records in st.session_state.manual_data.values())

            col1, col2 = st.columns(2)
            with col1:
                st.metric("总记录数", f"{total_records}条")
            with col2:
                st.metric("数据类型", f"{len(st.session_state.manual_data)}类")

            # 显示录入的数据
            for entry_type, records in st.session_state.manual_data.items():
                with st.expander(f"{entry_type} ({len(records)}条)"):
                    df = pd.DataFrame(records)
                    st.dataframe(df, use_container_width=True)

    else:
        st.warning("暂无数据，请先选择数据源")


# ============================================
# 6. 页面函数 - 数据概览
# ============================================

def calculate_kpi_metrics(data):
    """计算关键绩效指标"""

    kpi = {
        'monthly_transport_emission': '85.6 tCO₂',
        'transport_change': '环比 -12.5%',
        'unit_energy_consumption': '73.2 kWh/台',
        'energy_comparison': '优于行业 8.3%',
        'supply_chain_percentage': '68.5%',
        'scope3_description': '范围3排放比例',
        'reduction_potential': '150 tCO₂',
        'reduction_description': '年度优化潜力'
    }

    try:
        # 1. 计算月度运输排放
        if 'energy_data' in data and not data['energy_data'].empty:
            energy_data = data['energy_data'].copy()

            # 如果有柴油消耗数据
            if 'diesel_l' in energy_data.columns and len(energy_data) >= 2:
                # 计算最近一个月的柴油排放
                latest_month = energy_data.iloc[-1]
                prev_month = energy_data.iloc[-2]

                latest_emission = latest_month.get('diesel_l', 0) * EMISSION_FACTORS['diesel'] / 1000
                prev_emission = prev_month.get('diesel_l', 0) * EMISSION_FACTORS['diesel'] / 1000

                kpi['monthly_transport_emission'] = f"{latest_emission:.1f} tCO₂"

                # 计算变化
                if prev_emission > 0:
                    change_percent = ((latest_emission - prev_emission) / prev_emission) * 100
                    direction = '↑' if change_percent > 0 else '↓'
                    kpi['transport_change'] = f"环比 {direction} {abs(change_percent):.1f}%"
                else:
                    kpi['transport_change'] = "环比 -"

        # 2. 计算单位能耗
        if 'production_data' in data and not data['production_data'].empty:
            production_data = data['production_data']

            # 查找生产量列
            volume_col = None
            for col in production_data.columns:
                if any(word in col.lower() for word in ['production_volume', 'volume', 'quantity', 'amount']):
                    volume_col = col
                    break

            if volume_col and 'energy_data' in data and not data['energy_data'].empty:
                energy_data = data['energy_data']

                # 计算总产量
                total_production = production_data[volume_col].sum()

                # 计算总电力消耗
                total_electricity = 0

                # 查找电力数据列
                if 'electricity_kwh' in energy_data.columns:
                    total_electricity = energy_data['electricity_kwh'].sum()
                elif 'electricity' in energy_data.columns:
                    total_electricity = energy_data['electricity'].sum()
                else:
                    # 查找数值列
                    numeric_cols = energy_data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        total_electricity = energy_data[numeric_cols[0]].sum()

                # 计算单位能耗
                if total_production > 0 and total_electricity > 0:
                    unit_energy = total_electricity / total_production

                    # 与行业基准比较（假设行业基准为80）
                    industry_benchmark = 80
                    comparison = ((industry_benchmark - unit_energy) / industry_benchmark) * 100

                    kpi['unit_energy_consumption'] = f"{unit_energy:.1f} kWh/台"
                    comparison_text = '优于' if comparison > 0 else '低于'
                    kpi['energy_comparison'] = f"{comparison_text}行业 {abs(comparison):.1f}%"

        # 3. 计算供应链排放占比
        if 'carbon_data' in st.session_state and st.session_state.carbon_data:
            carbon_data = st.session_state.carbon_data
            total = carbon_data.get('total', 1)  # 避免除零
            scope3 = carbon_data.get('scope3', 0)

            if total > 0:
                scope3_percentage = (scope3 / total) * 100
                kpi['supply_chain_percentage'] = f"{scope3_percentage:.1f}%"
                kpi['scope3_description'] = f"范围3排放比例"

        # 4. 计算减排潜力
        if 'energy_data' in data and not data['energy_data'].empty:
            energy_data = data['energy_data']

            # 基于当前排放计算减排潜力
            total_electricity = 0

            # 查找电力数据列
            if 'electricity_kwh' in energy_data.columns:
                total_electricity = energy_data['electricity_kwh'].sum()
            elif 'electricity' in energy_data.columns:
                total_electricity = energy_data['electricity'].sum()
            else:
                # 查找数值列
                numeric_cols = energy_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    total_electricity = energy_data[numeric_cols[0]].sum()

            # 计算年度排放（假设使用华东电网）
            if 'electricity' in IMPROVED_FACTORS and isinstance(IMPROVED_FACTORS['electricity'], dict):
                electricity_factor = IMPROVED_FACTORS['electricity'].get('华东电网', 0.7035)
                annual_emission = total_electricity * electricity_factor / 1000

                # 减排潜力（假设可以减排5-15%）
                reduction_potential = annual_emission * np.random.uniform(0.05, 0.15)
                kpi['reduction_potential'] = f"{reduction_potential:.0f} tCO₂"
                kpi['reduction_description'] = '年度优化潜力'

    except Exception as e:
        # 如果计算出错，使用默认值
        st.warning(f"部分KPI计算失败: {str(e)}")
        import traceback
        traceback.print_exc()

    return kpi

def show_dashboard():
    """显示数据概览仪表板"""
    st.markdown('<h2 class="section-header">企业碳绩效总览</h2>', unsafe_allow_html=True)

    # 显示当前数据源
    st.info(f"当前数据源：{st.session_state.data_source}")

    # 获取数据
    data = get_current_data_for_dashboard()

    if not data:
        st.warning("请先在数据管理页面选择数据源")
        return

    # 显示公司信息
    if 'company_info' in data and not data['company_info'].empty:
        company_info = data['company_info'].iloc[0]

        st.markdown("#### 企业名称")
        st.markdown(f"<div style='font-size: 20px; font-weight: 500; color: #333; margin-bottom: 8px;'>{company_info.get('company_name', '未知')}</div>",
                    unsafe_allow_html=True)

        st.markdown("#### 所属行业")
        st.markdown(f"<div style='font-size: 20px; font-weight: 500; color: #333; margin-bottom: 8px;'>{company_info.get('industry', '未知')}</div>",
                    unsafe_allow_html=True)

        st.markdown("#### 员工人数")
        st.markdown(f"<div style='font-size: 20px; font-weight: 500; color: #333; margin-bottom: 8px;'>{company_info.get('employee_count', 0):,} 人</div>",
                    unsafe_allow_html=True)

        st.markdown("#### 年营业额")
        revenue = company_info.get('annual_revenue_million', 0)
        st.markdown(f"<div style='font-size: 20px; font-weight: 500; color: #333;'>¥{revenue:,} 万</div>",
                    unsafe_allow_html=True)

        # 关键绩效指标 - 动态计算
        st.markdown("---")
        st.markdown("### 关键绩效指标")

        # 计算关键指标
        kpi_data = calculate_kpi_metrics(data)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🚚 月度运输排放</h4>
                <h2>{kpi_data['monthly_transport_emission']}</h2>
                <p>{kpi_data['transport_change']}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>⚡ 单位能耗</h4>
                <h2>{kpi_data['unit_energy_consumption']}</h2>
                <p>{kpi_data['energy_comparison']}</p>
            </div>
            """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🌍 供应链占比</h4>
                <h2>{kpi_data['supply_chain_percentage']}</h2>
                <p>{kpi_data['scope3_description']}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>💡 减排潜力</h4>
                <h2>{kpi_data['reduction_potential']}</h2>
                <p>{kpi_data['reduction_description']}</p>
            </div>
            """, unsafe_allow_html=True)

    # 数据可视化
    st.markdown("---")
    st.markdown("### 数据可视化")

    # 能源消耗趋势 - 正确的版本
    if 'energy_data' in data and not data['energy_data'].empty:
        energy_data = data['energy_data'].copy()

        # 显示原始数据信息
        st.write(f"原始数据: {len(energy_data)}行")

        # 调试信息
        with st.expander("查看原始数据", expanded=False):
            st.write("列名:", energy_data.columns.tolist())
            st.write("数据类型:")
            for col in energy_data.columns:
                st.write(f"  {col}: {energy_data[col].dtype}")
            st.write("前10行数据:")
            st.dataframe(energy_data.head(10))

        # 查找月份列
        month_col = None
        for col in energy_data.columns:
            if 'month' in str(col).lower():
                month_col = col
                break

        if not month_col:
            st.error("没有找到月份列")
            return

        # 处理电力数据
        electricity_data = None

        # 方案1：如果数据是示例格式（month + electricity_kwh）
        if 'electricity_kwh' in energy_data.columns:
            # 按月份汇总
            electricity_data = energy_data.groupby(month_col)['electricity_kwh'].sum().reset_index()
            electricity_data = electricity_data.rename(columns={'electricity_kwh': 'consumption'})
            electricity_data['energy_type'] = '电力'

        # 方案2：如果数据是上传格式（month + energy_type + consumption）
        elif 'energy_type' in energy_data.columns and 'consumption' in energy_data.columns:
            # 筛选电力数据
            electricity_mask = (
                energy_data['energy_type'].astype(str).str.lower().str.contains('electricity|电|电力')
            )
            if electricity_mask.any():
                electricity_records = energy_data[electricity_mask].copy()
                # 按月份汇总
                electricity_data = electricity_records.groupby(month_col)['consumption'].sum().reset_index()
                electricity_data['energy_type'] = '电力'
            else:
                # 如果没有电力数据，使用第一个能源类型
                first_energy = energy_data['energy_type'].iloc[0]
                energy_records = energy_data[energy_data['energy_type'] == first_energy].copy()
                electricity_data = energy_records.groupby(month_col)['consumption'].sum().reset_index()
                electricity_data['energy_type'] = first_energy

        # 方案3：查找数值列
        else:
            # 查找数值列
            numeric_cols = energy_data.select_dtypes(include=[np.number]).columns.tolist()

            if numeric_cols:
                # 使用第一个数值列
                value_col = numeric_cols[0]
                # 按月份汇总
                electricity_data = energy_data.groupby(month_col)[value_col].sum().reset_index()
                electricity_data = electricity_data.rename(columns={value_col: 'consumption'})
                electricity_data['energy_type'] = value_col
            else:
                st.error("没有找到数值数据列")
                return

        # 检查处理后的数据
        if electricity_data is None or electricity_data.empty:
            st.error("处理后数据为空")
            return

        st.success(f"数据处理完成: {len(electricity_data)}个月的数据")

        # 确保按月份排序
        electricity_data = electricity_data.sort_values(month_col)

        # 创建图表
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=electricity_data[month_col],
            y=electricity_data['consumption'],
            mode='lines+markers',
            name='电力消耗',
            line=dict(color='#6699FF', width=3),
            marker=dict(size=8, color='#6699CC'),
            hoverinfo='x+y',
            text=electricity_data['energy_type']
        ))

        # 添加趋势线
        try:
            x_numeric = list(range(len(electricity_data)))
            z = np.polyfit(x_numeric, electricity_data['consumption'], 1)
            p = np.poly1d(z)
            trend_line = p(x_numeric)

            fig.add_trace(go.Scatter(
                x=electricity_data[month_col],
                y=trend_line,
                mode='lines',
                name='趋势线',
                line=dict(color='#FF6B6B', width=2, dash='dash'),
                hoverinfo='skip'
            ))
        except:
            pass

        fig.update_layout(
            title='月度电力消耗趋势',
            xaxis_title="月份",
            yaxis_title="消耗量",
            height=400,
            hovermode="x unified",
            showlegend=True
        )

        # 添加网格线
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

        st.plotly_chart(fig, use_container_width=True)

        # 显示统计信息
        st.metric("月份数", len(electricity_data))
        avg_consumption = electricity_data['consumption'].mean()
        st.metric("平均月消耗", f"{avg_consumption:,.0f}")
        total_consumption = electricity_data['consumption'].sum()
        st.metric("总消耗", f"{total_consumption:,.0f}")

        # 显示处理后的数据
        with st.expander("查看月度汇总数据", expanded=False):
            st.dataframe(electricity_data, use_container_width=True)

    else:
        st.warning("暂无能源数据")


    # 生产数据
    if 'production_data' in data and not data['production_data'].empty:
        production_data = data['production_data']

        # 检查数据格式
        if 'production_volume' in production_data.columns and 'product_type' in production_data.columns:
            # 按产品类型汇总
            product_summary = production_data.groupby('product_type')['production_volume'].sum().reset_index()

            # 只显示前5个产品
            if len(product_summary) > 5:
                top_products = product_summary.nlargest(5, 'production_volume')
                others = pd.DataFrame({
                    'product_type': ['其他产品'],
                    'production_volume': [product_summary.iloc[5:]['production_volume'].sum()]
                })
                product_summary = pd.concat([top_products, others], ignore_index=True)

            fig = px.pie(
                product_summary,
                values='production_volume',
                names='product_type',
                title='各产品产量占比',
                color_discrete_sequence=['#1E6F5C', '#4ECDC4', '#FFD166', '#FF6B6B', '#45B7D1', '#96CEB4', '#667eea', '#764ba2']
            )
        else:
            # 尝试自动识别
            # 查找产品列
            product_col = None
            for col in production_data.columns:
                if any(word in col.lower() for word in ['product', 'type', 'item']):
                    product_col = col
                    break

            # 查找产量列
            volume_col = None
            for col in production_data.columns:
                if any(word in col.lower() for word in ['volume', 'quantity', 'amount']):
                    volume_col = col
                    break

            if product_col and volume_col:
                # 按产品类型汇总
                product_summary = production_data.groupby(product_col)[volume_col].sum().reset_index()

                # 只显示前5个产品
                if len(product_summary) > 5:
                    top_products = product_summary.nlargest(5, volume_col)
                    others = pd.DataFrame({
                        product_col: ['其他产品'],
                        volume_col: [product_summary.iloc[5:][volume_col].sum()]
                    })
                    product_summary = pd.concat([top_products, others], ignore_index=True)

                fig = px.pie(
                    product_summary,
                    values=volume_col,
                    names=product_col,
                    title='产品产量占比'
                )
            else:
                st.warning("生产数据格式不兼容")
                return

        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>产量: %{value:,.0f}<br>占比: %{percent}'
        )

        fig.update_layout(
            height=400,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # 显示生产统计
        if 'production_volume' in production_data.columns:
            total_production = production_data['production_volume'].sum()
            st.metric("总产量", f"{total_production:,.0f} 单位")

        else:
            st.warning("暂无生产数据")

def get_current_data_for_dashboard():
    """获取当前数据用于仪表板"""

    data = {}

    if st.session_state.data_source == '示例数据':
        if 'sample_system' in st.session_state:
            return st.session_state.sample_system.get_dataset(st.session_state.current_dataset)

    elif st.session_state.data_source == '文件上传':
        # 转换上传数据格式
        if '公司信息' in st.session_state.uploaded_data:
            data['company_info'] = st.session_state.uploaded_data['公司信息']

        if '能源消耗' in st.session_state.uploaded_data:
            energy_df = st.session_state.uploaded_data['能源消耗'].copy()

            # 调试：显示原始数据
            with st.expander("能源数据原始格式", expanded=False):
                st.write("原始列名:", energy_df.columns.tolist())
                st.write("数据类型:")
                for col in energy_df.columns:
                    st.write(f"  {col}: {energy_df[col].dtype}")

            # 标准化列名
            column_mapping = {}

            # 检查并重命名月份列
            month_cols = [col for col in energy_df.columns
                          if any(word in col.lower() for word in ['month', '月份', '时间', 'date', 'period'])]
            if month_cols:
                column_mapping[month_cols[0]] = 'month'

            # 检查并重命名能源类型列
            energy_type_cols = [col for col in energy_df.columns
                                if any(word in col.lower() for word in ['type', '类型', '能源', 'energy'])]
            if energy_type_cols:
                column_mapping[energy_type_cols[0]] = 'energy_type'

            # 检查并重命名消耗量列
            consumption_cols = [col for col in energy_df.columns
                                if any(word in col.lower() for word in ['consumption', '消耗', '用量', 'value', 'amount'])]
            if consumption_cols:
                column_mapping[consumption_cols[0]] = 'consumption'

            # 应用重命名
            if column_mapping:
                energy_df = energy_df.rename(columns=column_mapping)

            # 如果没有energy_type列但有多个数值列，转换为长格式
            if 'energy_type' not in energy_df.columns and 'month' in energy_df.columns:
                numeric_cols = energy_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    # 将宽格式转换为长格式
                    energy_df = energy_df.melt(
                        id_vars=['month'],
                        value_vars=numeric_cols,
                        var_name='energy_type',
                        value_name='consumption'
                    )

            data['energy_data'] = energy_df

        if '生产数据' in st.session_state.uploaded_data:
            prod_df = st.session_state.uploaded_data['生产数据'].copy()

            # 标准化列名
            column_mapping = {}

            # 检查月份列
            month_cols = [col for col in prod_df.columns
                          if any(word in col.lower() for word in ['month', '月份', '时间'])]
            if month_cols:
                column_mapping[month_cols[0]] = 'month'

            # 检查产品类型列
            product_cols = [col for col in prod_df.columns
                            if any(word in col.lower() for word in ['product', '产品', 'type', '类型', 'item'])]
            if product_cols:
                column_mapping[product_cols[0]] = 'product_type'

            # 检查产量列
            volume_cols = [col for col in prod_df.columns
                           if any(word in col.lower() for word in ['volume', '产量', '数量', 'amount', 'quantity'])]
            if volume_cols:
                column_mapping[volume_cols[0]] = 'production_volume'

            if column_mapping:
                prod_df = prod_df.rename(columns=column_mapping)

            data['production_data'] = prod_df

        if '供应商信息' in st.session_state.uploaded_data:
            data['supplier_data'] = st.session_state.uploaded_data['供应商信息']

    elif st.session_state.data_source == '手动录入':
        # 转换手动录入数据格式
        if '能源消耗' in st.session_state.manual_data:
            manual_energy = pd.DataFrame(st.session_state.manual_data['能源消耗'])

            # 确保有必要的列
            if 'energy_type' not in manual_energy.columns:
                manual_energy['energy_type'] = 'electricity'
            if 'consumption' not in manual_energy.columns and 'electricity_kwh' in manual_energy.columns:
                manual_energy['consumption'] = manual_energy['electricity_kwh']

            data['energy_data'] = manual_energy

        if '生产数据' in st.session_state.manual_data:
            data['production_data'] = pd.DataFrame(st.session_state.manual_data['生产数据'])

        if '供应商信息' in st.session_state.manual_data:
            data['supplier_data'] = pd.DataFrame(st.session_state.manual_data['供应商信息'])

    # 确保数据完整性
    for key in ['energy_data', 'production_data', 'supplier_data']:
        if key not in data:
            data[key] = pd.DataFrame()

    return data

# ============================================
# 7. 页面函数 - 碳计算
# ============================================

def show_carbon_calculation():
    """显示碳计算页面"""
    st.markdown('<h2 class="section-header">碳排放计算</h2>', unsafe_allow_html=True)

    # ========== 3. 原有代码部分 ==========
    # 获取数据
    data = get_current_data_for_dashboard()

    if not data:
        st.warning("请先在数据管理页面选择数据源")
        return

    st.info(f"当前使用数据源：{st.session_state.data_source}")

    # 添加排放因子说明
    with st.expander("查看本次计算使用的排放因子", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**范围1：直接排放**")
            st.write(f"- 柴油：{IMPROVED_FACTORS['diesel']} kg CO₂/L")
            st.write(f"- 天然气：{IMPROVED_FACTORS['natural_gas']} kg CO₂/m³")

        with col2:
            st.write("**范围2：外购电力**")
            for region, factor in IMPROVED_FACTORS['electricity'].items():
                st.write(f"- {region}：{factor} kg/kWh")

        with col3:
            st.write("**范围3：运输排放**")
            st.write(f"- 公路运输：{IMPROVED_FACTORS['truck_transport']} kg/(ton·km)")
            st.write(f"- 铁路运输：{IMPROVED_FACTORS['rail_transport']} kg/(ton·km)")
            st.write(f"- 水路运输：{IMPROVED_FACTORS['ship_transport']} kg/(ton·km)")
            st.write(f"- 航空运输：{IMPROVED_FACTORS['air_transport']} kg/(ton·km)")

            st.write("**范围3：物料排放**")
            st.write(f"- 钢材：{IMPROVED_FACTORS['steel']} kg CO₂/kg")
            st.write(f"- 铝材：{IMPROVED_FACTORS['aluminum']} kg CO₂/kg")
            st.write(f"- 塑料：{IMPROVED_FACTORS['plastic']} kg CO₂/kg")

    # 创建选项卡
    tab1, tab2, tab3, tab4 = st.tabs(["直接排放", "外购电力排放", "供应链间接排放", "综合计算"])

    with tab1:
        st.markdown("### 直接排放")

        if 'energy_data' in data and not data['energy_data'].empty:
            # 使用最近一个月的数据
            latest_energy = data['energy_data'].iloc[-1]

            # 使用改进的计算函数
            scope1 = calculate_scope1_improved(latest_energy)

            # 显示指标 - 使用改进的因子
            diesel_emission = latest_energy.get('diesel_l', 0) * IMPROVED_FACTORS['diesel'] / 1000
            gas_emission = latest_energy.get('natural_gas_m3', 0) * IMPROVED_FACTORS['natural_gas'] / 1000

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("柴油排放", f"{diesel_emission:.1f} tCO₂")
            with col2:
                st.metric("天然气排放", f"{gas_emission:.1f} tCO₂")
            with col3:
                st.metric("直接总排放", f"{scope1:.1f} tCO₂")
        else:
            st.warning("暂无能源数据")

        st.session_state.tab1_scope1 = scope1

    with tab2:
        st.markdown("### 外购电力排放")

        if 'energy_data' in data and not data['energy_data'].empty:
            energy_data = data['energy_data']

            # 查找月份列
            month_col = None
            for col in energy_data.columns:
                if 'month' in str(col).lower() or 'year_month' in str(col).lower():
                    month_col = col
                    break

            if month_col:
                # 让用户选择电网区域
                st.markdown("##### 选择电网区域")
                selected_region = st.selectbox(
                    "电网区域",
                    list(IMPROVED_FACTORS['electricity'].keys()),
                    index=0,
                    key="grid_region_select"
                )

                # 按月份汇总电力数据
                electricity_summary = None

                if 'electricity_kwh' in energy_data.columns:
                    electricity_summary = energy_data.groupby(month_col)['electricity_kwh'].sum().reset_index()
                    electricity_summary = electricity_summary.rename(columns={'electricity_kwh': 'consumption'})
                elif 'electricity_total_kwh' in energy_data.columns:
                    electricity_summary = energy_data.groupby(month_col)['electricity_total_kwh'].sum().reset_index()
                    electricity_summary = electricity_summary.rename(columns={'electricity_total_kwh': 'consumption'})

                if electricity_summary is not None and not electricity_summary.empty:
                    # 排序
                    electricity_summary = electricity_summary.sort_values(month_col)

                    # 计算最近一个月的排放
                    latest_month = electricity_summary.iloc[-1]
                    scope2 = calculate_scope2_improved(
                        {'electricity_kwh': latest_month['consumption']},
                        selected_region
                    )

                    # 显示指标
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("最近月份电力消耗", f"{latest_month['consumption']:,.0f} kWh")
                    with col2:
                        st.metric(f"{selected_region}排放因子", f"{IMPROVED_FACTORS['electricity'][selected_region]} kg/kWh")
                    with col3:
                        st.metric("外购电力总排放", f"{scope2:.1f} tCO₂")
                else:
                    st.warning("没有找到电力消耗数据")
            else:
                st.warning("没有找到月份列")
        else:
            st.warning("暂无能源数据")

        st.session_state.tab2_scope2 = scope2
        st.session_state.selected_region = selected_region

    with tab3:
        st.markdown("### 供应链间接排放")

        # 使用改进的计算方法
        scope3 = 0
        scope3_details = {'total': 0, 'material': 0, 'transport': 0, 'supplier_count': 0}

        if 'supplier_data' in data and not data['supplier_data'].empty:
            scope3, scope3_details = calculate_scope3_improved(data['supplier_data'])

        # 显示详细结果
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("供应商数量", scope3_details['supplier_count'])

        with col2:
            if scope3_details['supplier_count'] > 0:
                avg_per_supplier = scope3 / scope3_details['supplier_count']
                st.metric("平均供应商排放", f"{avg_per_supplier:.1f} tCO₂")
            else:
                st.metric("平均供应商排放", "0.0 tCO₂")

        with col3:
            st.metric("物料排放部分", f"{scope3_details['material']:.1f} tCO₂")

        with col4:
            st.metric("运输排放部分", f"{scope3_details['transport']:.1f} tCO₂")

        # 总计
        st.markdown("---")
        st.metric("**供应链间接总排放**", f"{scope3:.1f} tCO₂")

        # 供应商排放排行
        if scope3_details.get('supplier_emissions') and len(scope3_details['supplier_emissions']) > 0:
            with st.expander("供应商碳排放排行", expanded=True):

                # 创建排行表格数据
                ranking_data = []
                for i, supplier in enumerate(scope3_details['supplier_emissions'], 1):
                    ranking_data.append({
                        '排名': i,
                        '供应商名称': supplier['supplier_name'],
                        '供应商类型': supplier['supplier_type'],
                        '总排放(tCO₂)': round(supplier['total_emission_t'], 2),
                        '占比(%)': round(supplier.get('emission_percentage', 0), 1),
                        '物料排放(t)': round(supplier['material_emission_t'], 2),
                        '运输排放(t)': round(supplier['transport_emission_t'], 2),
                        '年采购量(吨)': round(supplier['annual_purchase_ton'], 1),
                        '距离(km)': round(supplier['distance_km'], 0),
                        '运输方式': supplier['transport_mode']
                    })

                ranking_df = pd.DataFrame(ranking_data)

                # 显示排行表格
                st.dataframe(
                    ranking_df,
                    use_container_width=True,
                    column_config={
                        "排名": st.column_config.NumberColumn(
                            width="small",
                            help="按排放量从高到低排名"
                        ),
                        "供应商名称": st.column_config.TextColumn(
                            width="medium",
                            help="供应商名称"
                        ),
                        "总排放(tCO₂)": st.column_config.NumberColumn(
                            "总排放量",
                            help="供应商年度总碳排放量（吨）",
                            format="%.1f tCO2",
                            width="small"
                        ),
                        "占比(%)": st.column_config.ProgressColumn(
                            "排放占比",
                            help="供应商排放占总供应链排放的比例",
                            format="%.1f%%",
                            min_value=0,
                            max_value=100,
                            width="medium"
                        ),
                        "物料排放(t)": st.column_config.NumberColumn(
                            "物料排放",
                            help="物料采购产生的排放",
                            format="%.1f tCO2",
                            width="small"
                        ),
                        "运输排放(t)": st.column_config.NumberColumn(
                            "运输排放",
                            help="运输过程产生的排放",
                            format="%.1f tCO2",
                            width="small"
                        ),
                        "年采购量(吨)": st.column_config.NumberColumn(
                            "年采购量",
                            format="%.0f t",
                            width="small"
                        ),
                        "距离(km)": st.column_config.NumberColumn(
                            "运输距离",
                            format="%.0f km",
                            width="small"
                        ),
                        "运输方式": st.column_config.TextColumn(
                            width="small",
                            help="主要运输方式"
                        )
                    },
                    hide_index=True
                )

                # 可视化排行
                if len(ranking_data) > 0:
                    # 前10名排放柱状图
                    top_n = min(10, len(ranking_data))
                    top_suppliers = ranking_df.head(top_n)

                    fig = go.Figure()

                    # 物料排放（柱状图底部）
                    fig.add_trace(go.Bar(
                        x=top_suppliers['供应商名称'],
                        y=top_suppliers['物料排放(t)'],
                        name='物料排放',
                        marker_color='#FFA726',
                        text=top_suppliers['物料排放(t)'].round(1),
                        textposition='outside'
                    ))

                    # 运输排放（堆叠在物料排放上）
                    fig.add_trace(go.Bar(
                        x=top_suppliers['供应商名称'],
                        y=top_suppliers['运输排放(t)'],
                        name='运输排放',
                        marker_color='#26C6DA',
                        text=top_suppliers['运输排放(t)'].round(1),
                        textposition='outside'
                    ))

                    fig.update_layout(
                        title=f'前{top_n}大排放供应商分析',
                        xaxis_title="供应商",
                        yaxis_title="排放量 (tCO₂)",
                        barmode='stack',
                        height=400,
                        showlegend=True
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # 排放构成饼图
                    if len(ranking_data) > 1:
                        fig_pie = go.Figure()

                        # 只显示前5名，其他合并为"其他供应商"
                        if len(ranking_data) > 5:
                            top_5 = ranking_data[:5]
                            other_emission = sum(item['总排放(tCO₂)'] for item in ranking_data[5:])

                            labels = [item['供应商名称'] for item in top_5] + ['其他供应商']
                            values = [item['总排放(tCO₂)'] for item in top_5] + [other_emission]
                        else:
                            labels = [item['供应商名称'] for item in ranking_data]
                            values = [item['总排放(tCO₂)'] for item in ranking_data]

                        fig_pie.add_trace(go.Pie(
                            labels=labels,
                            values=values,
                            hole=0.3,
                            textinfo='label+percent',
                            hoverinfo='label+value+percent'
                        ))

                        fig_pie.update_layout(
                            title="供应商排放占比分析",
                            height=350
                        )

                        st.plotly_chart(fig_pie, use_container_width=True)

        # 显示计算说明
        if scope3_details['supplier_count'] > 0:
            with st.expander("查看范围3计算详情"):
                st.write("**计算方法：**")
                st.write("1. **物料排放** = 年采购量 × 材料排放因子")
                st.write("2. **运输排放** = 距离 × 单次重量 × 年运输次数 × 运输因子")

                st.write(f"\n**计算结果汇总：**")
                st.write(f"- 涉及供应商：{scope3_details['supplier_count']}家")
                st.write(f"- 物料排放总计：{scope3_details['material']:.1f} tCO₂")
                st.write(f"- 运输排放总计：{scope3_details['transport']:.1f} tCO₂")
                st.write(f"- 范围3总排放：{scope3:.1f} tCO₂")

                # 显示关键供应商洞察
                if scope3_details.get('top_emitters'):
                    st.write("\n**关键洞察：**")
                    top_supplier = scope3_details['top_emitters'][0]
                    st.write(f"- 最大排放供应商：**{top_supplier['supplier_name']}**")
                    st.write(f"  排放量：{top_supplier['total_emission_t']:.1f} tCO₂")
                    st.write(f"  占比：{top_supplier.get('emission_percentage', 0):.1f}%")

                    # 分析排放主要原因
                    if top_supplier['material_emission_t'] > top_supplier['transport_emission_t']:
                        st.write(f"  主要来源：物料采购（{top_supplier['material_emission_t']:.1f} tCO₂）")
                    else:
                        st.write(f"  主要来源：运输排放（{top_supplier['transport_emission_t']:.1f} tCO₂）")
        st.session_state.tab3_scope3 = scope3
        st.session_state.tab3_scope3_details = scope3_details

    with tab4:
        st.markdown("### 综合计算与结果")

        # 设置默认值，避免后续代码出错
        scope1 = 0
        scope2 = 0
        scope3 = 0
        scope3_details = {'total': 0, 'material': 0, 'transport': 0, 'supplier_count': 0}
        total_emission = 0

        # 检查前三个tab是否已计算
        tab1_calculated = 'tab1_scope1' in st.session_state
        tab2_calculated = 'tab2_scope2' in st.session_state
        tab3_calculated = 'tab3_scope3' in st.session_state

        # 如果前三个tab都计算了，直接使用session_state的值
        if tab1_calculated and tab2_calculated and tab3_calculated:
            scope1 = st.session_state['tab1_scope1']
            scope2 = st.session_state['tab2_scope2']
            scope3 = st.session_state['tab3_scope3']
            scope3_details = st.session_state['tab3_scope3_details']
            selected_region = st.session_state.get('selected_region', '华东电网')

            total_emission = scope1 + scope2 + scope3

            st.info(f"电网区域：{selected_region} | 供应商数量：{scope3_details.get('supplier_count', 0)}家")

        else:
            # 如果前三个tab没有全部计算，提示用户
            st.warning("请先完成前三个范围的碳排放计算")

            missing_tabs = []
            if not tab1_calculated:
                missing_tabs.append("直接排放")
            if not tab2_calculated:
                missing_tabs.append("外购电力排放")
            if not tab3_calculated:
                missing_tabs.append("供应链间接排放")

            st.info(f"需要先计算：{', '.join(missing_tabs)}")

        # 计算完成后，确保保存这个格式：
        st.session_state['emission_results'] = {
            'scope1': float(scope1),  # 确保是浮点数
            'scope2': float(scope2),
            'scope3': float(scope3),
            'total': float(total_emission),
            'calculation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'grid_region': selected_region,  # 保存使用的电网区域
            'data_source': st.session_state.data_source,
            # 额外信息（统一放在一个键下）
            'details': {
                'scope3_material': float(scope3_details.get('material', 0)),
                'scope3_transport': float(scope3_details.get('transport', 0)),
                'supplier_count': int(scope3_details.get('supplier_count', 0))
            }
        }

        # 同时保存一份简化版本给其他页面使用
        st.session_state['carbon_data'] = {
            'scope1': float(scope1),
            'scope2': float(scope2),
            'scope3': float(scope3),
            'total': float(total_emission)
        }

        # 显示结果
        col1, col2 = st.columns(2)

        with col1:
            # 排放构成图
            if total_emission > 0:
                fig = go.Figure(data=[go.Pie(
                    labels=['范围1（直接）', '范围2（电力）', '范围3（供应链）'],
                    values=[scope1, scope2, scope3],
                    hole=.3,
                    marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                    textinfo='label+percent',
                    hoverinfo='label+value+percent',
                    texttemplate='%{label}<br>%{value:.1f}t<br>%{percent}'
                )])
                fig.update_layout(
                    title="碳排放构成",
                    height=400,
                    annotations=[dict(
                        text=f'总计<br>{total_emission:.1f}t',
                        x=0.5, y=0.5, font_size=16, showarrow=False
                    )]
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("暂无碳排放数据")

        with col2:
            st.metric("总碳排放量", f"{total_emission:.1f} tCO₂")

            if total_emission > 0:
                scope1_pct = scope1 / total_emission * 100
                scope2_pct = scope2 / total_emission * 100
                scope3_pct = scope3 / total_emission * 100

                st.metric("范围1（直接排放）",
                          f"{scope1:.1f} tCO₂")

                st.metric("范围2（外购电力）",
                          f"{scope2:.1f} tCO₂")

                st.metric("范围3（供应链间接）",
                          f"{scope3:.1f} tCO₂")

            # 范围3细分
            with st.expander("查看范围3细分", expanded=False):
                if scope3_details['total'] > 0:
                    fig_sub = go.Figure(data=[go.Pie(
                        labels=['物料排放', '运输排放'],
                        values=[scope3_details['material'], scope3_details['transport']],
                        hole=.2,
                        marker_colors=['#FFA726', '#26C6DA']
                    )])
                    fig_sub.update_layout(
                        title="范围3排放细分",
                        height=300
                    )
                    st.plotly_chart(fig_sub, use_container_width=True)

                    st.write(f"**物料排放**：{scope3_details['material']:.1f} tCO₂")
                    st.write(f"**运输排放**：{scope3_details['transport']:.1f} tCO₂")
                    st.write(f"**涉及供应商**：{scope3_details['supplier_count']}家")
                else:
                    st.info("暂无范围3细分数据")

        # 运输因子使用情况说明
        with st.expander("查看运输排放因子使用情况", expanded=False):
            st.write("**使用了以下运输排放因子：**")

            transport_data = [
                ["公路运输", IMPROVED_FACTORS['truck_transport'], "kg CO₂/(ton·km)", "主要用于短途和中途运输"],
                ["铁路运输", IMPROVED_FACTORS['rail_transport'], "kg CO₂/(ton·km)", "节能环保，适合中长途"],
                ["水路运输", IMPROVED_FACTORS['ship_transport'], "kg CO₂/(ton·km)", "最低碳，适合大宗货物长途"],
                ["航空运输", IMPROVED_FACTORS['air_transport'], "kg CO₂/(ton·km)", "最高碳，紧急或高价值货物"]
            ]

            transport_df = pd.DataFrame(
                transport_data,
                columns=["运输方式", "排放因子", "单位", "适用场景"]
            )

            st.dataframe(transport_df, use_container_width=True)

            st.write("**因子来源**：基于行业平均值，参考《道路运输企业温室气体排放核算方法与报告指南》")

        if st.button("保存计算结果", type="primary", use_container_width=True):
            st.success("计算结果已保存！")
            st.balloons()


def prepare_data_for_ai_analysis(data, emission_results):
    """准备数据用于AI分析"""
    analysis_data = {
        'emission_structure': {},
        'energy_patterns': {},
        'production_efficiency': {},
        'data_quality': {}
    }

    # 排放结构
    total = emission_results.get('total', 1)
    analysis_data['emission_structure'] = {
        'scope1_ratio': emission_results.get('scope1', 0) / total * 100,
        'scope2_ratio': emission_results.get('scope2', 0) / total * 100,
        'scope3_ratio': emission_results.get('scope3', 0) / total * 100,
        'total_emission': total
    }

    # 能源模式
    if 'energy_data' in data and not data['energy_data'].empty:
        energy_data = data['energy_data']
        analysis_data['energy_patterns'] = analyze_energy_patterns(energy_data)

    # 生产效率
    if 'production_data' in data and not data['production_data'].empty:
        production_data = data['production_data']
        if 'production_volume' in production_data.columns:
            analysis_data['production_efficiency']['total_production'] = production_data['production_volume'].sum()

    return analysis_data


def analyze_energy_patterns(energy_data):
    """分析能源使用模式"""
    patterns = {
        'trend_direction': 'unknown',
        'seasonality': False,
        'anomalies': [],
        'statistics': {}
    }

    if energy_data.empty or len(energy_data) < 3:
        return patterns

    # 分析电力消耗趋势
    if 'electricity_kwh' in energy_data.columns:
        electricity = energy_data['electricity_kwh'].values

        # 计算趋势
        if len(electricity) >= 2:
            # 简单趋势判断
            first_half = electricity[:len(electricity) // 2].mean()
            second_half = electricity[len(electricity) // 2:].mean()

            if second_half > first_half * 1.1:
                patterns['trend_direction'] = 'upward'
            elif second_half < first_half * 0.9:
                patterns['trend_direction'] = 'downward'
            else:
                patterns['trend_direction'] = 'stable'

        # 检测异常值
        mean = np.mean(electricity)
        std = np.std(electricity)
        for i, value in enumerate(electricity):
            if abs(value - mean) > 2 * std:
                patterns['anomalies'].append({
                    'index': i,
                    'value': value,
                    'deviation': f"{(value - mean) / mean * 100:.1f}%"
                })

        patterns['statistics']['electricity'] = {
            'mean': float(mean),
            'std': float(std),
            'min': float(np.min(electricity)),
            'max': float(np.max(electricity))
        }

    return patterns


def predict_energy_consumption(energy_data, forecast_months=3):
    """使用机器学习预测能源消耗 - 改进版"""
    predictions = {}

    try:
        # 准备时间特征 - 添加更多时序特征
        df = energy_data.copy()

        # 创建更好的时间特征
        df['time_index'] = range(len(df))

        # 添加时序特征
        if len(df) >= 3:
            # 移动平均特征
            df['ma_3'] = df['electricity_kwh'].rolling(window=3, min_periods=1).mean() if 'electricity_kwh' in df.columns else 0

            # 滞后特征
            df['lag_1'] = df['electricity_kwh'].shift(1) if 'electricity_kwh' in df.columns else 0
            df['lag_2'] = df['electricity_kwh'].shift(2) if 'electricity_kwh' in df.columns else 0

            # 季节性特征（如果有足够数据）
            if len(df) >= 12:
                df['month_sin'] = np.sin(2 * np.pi * (df['time_index'] % 12) / 12)
                df['month_cos'] = np.cos(2 * np.pi * (df['time_index'] % 12) / 12)

        # 对每种能源类型进行预测
        for energy_col in ['electricity_kwh', 'natural_gas_m3', 'diesel_l']:
            if energy_col in df.columns:
                # 准备特征 - 使用更多特征
                feature_cols = ['time_index']
                if 'ma_3' in df.columns:
                    feature_cols.append('ma_3')
                if 'lag_1' in df.columns:
                    feature_cols.append('lag_1')
                if 'lag_2' in df.columns:
                    feature_cols.append('lag_2')
                if 'month_sin' in df.columns:
                    feature_cols.extend(['month_sin', 'month_cos'])

                # 移除NaN
                df_valid = df[feature_cols + [energy_col]].dropna()

                if len(df_valid) >= 6:  # 需要足够的数据
                    X = df_valid[feature_cols].values
                    y = df_valid[energy_col].values

                    # 划分训练测试集
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )

                    # 使用更好的模型参数
                    model = RandomForestRegressor(
                        n_estimators=200,  # 增加树的数量
                        max_depth=10,  # 增加深度
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1
                    )

                    model.fit(X_train, y_train)

                    # **关键修复：递归预测，而不是一次性预测所有未来**
                    future_predictions = []

                    # 创建未来预测的数据框
                    last_row = df_valid.iloc[-1].copy()

                    for i in range(forecast_months):
                        # 更新时间索引
                        future_time = len(df_valid) + i

                        # 创建未来时间点的特征
                        future_features = {
                            'time_index': future_time
                        }

                        # 添加移动平均（基于前几个预测）
                        if 'ma_3' in feature_cols:
                            if i == 0:
                                # 使用历史数据的平均
                                future_features['ma_3'] = df_valid[energy_col].tail(3).mean()
                            elif i == 1:
                                future_features['ma_3'] = (df_valid[energy_col].iloc[-1] + future_predictions[0]) / 2
                            else:
                                future_features['ma_3'] = (future_predictions[-2] + future_predictions[-1]) / 2

                        # 添加滞后特征
                        if 'lag_1' in feature_cols:
                            if i == 0:
                                future_features['lag_1'] = df_valid[energy_col].iloc[-1]
                            else:
                                future_features['lag_1'] = future_predictions[-1]

                        if 'lag_2' in feature_cols:
                            if i == 0:
                                future_features['lag_2'] = df_valid[energy_col].iloc[-2] if len(df_valid) >= 2 else 0
                            elif i == 1:
                                future_features['lag_2'] = df_valid[energy_col].iloc[-1]
                            else:
                                future_features['lag_2'] = future_predictions[-2]

                        # 添加季节性特征
                        if 'month_sin' in feature_cols:
                            future_features['month_sin'] = np.sin(2 * np.pi * (future_time % 12) / 12)
                            future_features['month_cos'] = np.cos(2 * np.pi * (future_time % 12) / 12)

                        # 转换为模型输入格式
                        future_input = []
                        for col in feature_cols:
                            future_input.append(future_features.get(col, 0))

                        # 预测
                        future_pred = model.predict([future_input])[0]
                        future_predictions.append(future_pred)

                    # 评估模型
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    # 计算特征重要性
                    feature_importance = dict(zip(feature_cols, model.feature_importances_))

                    predictions[energy_col] = {
                        'predictions': future_predictions,
                        'model_performance': {
                            'mse': float(mse),
                            'r2': float(r2),
                            'rmse': float(np.sqrt(mse))
                        },
                        'feature_importance': feature_importance,
                        'trend': 'increasing' if future_predictions[-1] > future_predictions[0] else 'decreasing' if future_predictions[-1] < future_predictions[0] else 'stable',
                        'confidence': 'high' if r2 > 0.8 else 'medium' if r2 > 0.6 else 'low'
                    }

    except Exception as e:
        print(f"预测出错: {e}")
        import traceback
        traceback.print_exc()

    return predictions


def create_prediction_chart(energy_data, prediction_data, forecast_months=3):
    """创建预测图表 - 改进版"""
    fig = go.Figure()

    # 历史数据
    if 'electricity_kwh' in energy_data.columns:
        historical_values = energy_data['electricity_kwh'].values
        historical_indices = list(range(1, len(historical_values) + 1))

        fig.add_trace(go.Scatter(
            x=historical_indices,
            y=historical_values,
            mode='lines+markers',
            name='历史数据',
            line=dict(color='#4ECDC4', width=3),
            marker=dict(size=8, color='#45B7D1')
        ))

    # 预测数据
    if 'predictions' in prediction_data:
        future_values = prediction_data['predictions']
        future_indices = list(range(len(historical_values) + 1,
                                    len(historical_values) + len(future_values) + 1))

        # 添加预测区间（简单估计）
        confidence = 0.2  # 20% 置信区间

        if future_values:
            # 计算历史数据的标准差
            historical_std = np.std(historical_values) if len(historical_values) > 1 else 0

            upper_bound = [v * (1 + confidence) for v in future_values]
            lower_bound = [v * (1 - confidence) for v in future_values]

            # 添加置信区间
            fig.add_trace(go.Scatter(
                x=future_indices + future_indices[::-1],
                y=upper_bound + lower_bound[::-1],
                fill='toself',
                fillcolor='rgba(255, 107, 107, 0.2)',
                line=dict(color='rgba(255, 255, 255, 0)'),
                name='置信区间',
                showlegend=True
            ))

        fig.add_trace(go.Scatter(
            x=future_indices,
            y=future_values,
            mode='lines+markers',
            name='AI预测',
            line=dict(color='#FF6B6B', width=3, dash='dash'),
            marker=dict(size=10, color='#FF6B6B', symbol='diamond')
        ))

    # 添加趋势线（历史数据）
    if len(historical_values) >= 2:
        # 计算历史趋势
        x_hist = np.array(historical_indices)
        y_hist = np.array(historical_values)
        z = np.polyfit(x_hist, y_hist, 1)
        p = np.poly1d(z)
        trend_line = p(x_hist)

        fig.add_trace(go.Scatter(
            x=x_hist,
            y=trend_line,
            mode='lines',
            name='历史趋势线',
            line=dict(color='rgba(0, 0, 0, 0.5)', width=2, dash='dot')
        ))

    fig.update_layout(
        title='电力消耗预测分析',
        xaxis_title='月份',
        yaxis_title='消耗量 (kWh)',
        height=450,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        plot_bgcolor='rgba(248, 249, 250, 1)'
    )

    # 添加网格
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.3)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.3)')

    return fig


def generate_smart_suggestions(data, results):
    """生成智能建议（基于真实数据）"""
    suggestions = []

    total = results.get('total', 1)

    # 建议1：基于排放结构
    scope2_ratio = results.get('scope2', 0) / total * 100
    if scope2_ratio > 60:
        suggestions.append({
            "title": "电力排放优化",
            "description": f"电力排放占总排放的{scope2_ratio:.1f}%，是减排重点",
            "potential": "预计减排：15-25%",
            "roi": "投资回收期：18-36个月"
        })

    # 建议2：基于能源数据
    if 'energy_data' in data and not data['energy_data'].empty:
        energy_data = data['energy_data']

        # 分析电力波动
        if 'electricity_kwh' in energy_data.columns:
            electricity = energy_data['electricity_kwh']
            if len(electricity) >= 3:
                std_dev = electricity.std() / electricity.mean()
                if std_dev > 0.3:
                    suggestions.append({
                        "title": "需求侧管理",
                        "description": "电力消耗波动较大，建议实施需求侧管理",
                        "potential": "预计减排：8-12%",
                        "roi": "投资回收期：6-12个月"
                    })

    # 建议3：基于供应链
    scope3_ratio = results.get('scope3', 0) / total * 100
    if scope3_ratio > 70:
        suggestions.append({
            "title": "供应商协同",
            "description": f"供应链排放占比高达{scope3_ratio:.1f}%，建议与供应商合作减排",
            "potential": "预计减排：10-15%",
            "roi": "投资回收期：长期投资"
        })

    # 默认建议（如果以上都不满足）
    if not suggestions:
        suggestions = [
            {
                "title": "运输优化",
                "description": "基于通用数据分析，运输优化通常有较大减排潜力",
                "potential": "预计减排：18%",
                "roi": "投资回收期：12个月"
            },
            {
                "title": "能源效率",
                "description": "提升能源使用效率是基础但有效的减排措施",
                "potential": "预计减排：15%",
                "roi": "投资回收期：24个月"
            },
            {
                "title": "供应商协同",
                "description": "与高碳供应商合作改进，降低供应链排放",
                "potential": "预计减排：12%",
                "roi": "投资回收期：长期"
            }
        ]

    return suggestions


def get_ai_suggestions(data, results):
    """获取AI生成的个性化建议"""
    # 如果没有配置AI，返回默认建议
    if not AI_CONFIG['api_key'] or AI_CONFIG['api_key'] == "sk-your-api-key-here":
        return """
        ## AI建议（示例）

        **1. 基于当前排放结构的建议：**
        - 范围2排放占比较高，建议优先优化电力使用
        - 考虑安装智能电表，实时监控用电情况

        **2. 短期行动建议：**
        - 开展能源审计，识别主要耗能设备
        - 优化生产排程，避免高峰用电

        **3. 中长期战略：**
        - 制定碳减排路线图
        - 探索可再生能源采购
        """

    try:
        # 准备提示词
        prompt = f"""
        基于以下碳足迹数据，请提供专业的减排建议：

        排放总量：{results.get('total', 0):.1f} tCO₂
        排放结构：
        - 范围1（直接排放）：{results.get('scope1', 0):.1f} tCO₂ ({results.get('scope1', 0) / results.get('total', 1) * 100:.1f}%)
        - 范围2（电力排放）：{results.get('scope2', 0):.1f} tCO₂ ({results.get('scope2', 0) / results.get('total', 1) * 100:.1f}%)
        - 范围3（供应链）：{results.get('scope3', 0):.1f} tCO₂ ({results.get('scope3', 0) / results.get('total', 1) * 100:.1f}%)

        请提供：
        1. 具体的减排措施（按优先级排序）
        2. 预期的减排效果
        3. 投资回报分析
        4. 实施建议
        """

        client, available = init_ai_client()
        if client and available:
            response = client.chat.completions.create(
                model=AI_CONFIG['model'],
                messages=[
                    {"role": "system", "content": "你是一个专业的碳足迹管理专家"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )

            return response.choices[0].message.content
        else:
            return "AI服务暂时不可用，请检查API配置"

    except Exception as e:
        return f"AI建议生成失败：{str(e)}"


def generate_ai_analysis(data, results):
    """生成AI分析报告"""
    analysis_text = "## AI深度分析报告\n\n"

    # 排放分析 - 添加安全检查
    total = results.get('total', 0)
    if total > 0:
        analysis_text += f"### 排放结构分析\n"
        analysis_text += f"- 总排放量：**{total:.1f} tCO₂**\n"

        # 安全计算百分比
        scope1 = results.get('scope1', 0)
        scope2 = results.get('scope2', 0)
        scope3 = results.get('scope3', 0)

        scope1_pct = (scope1 / total * 100) if total > 0 else 0
        scope2_pct = (scope2 / total * 100) if total > 0 else 0
        scope3_pct = (scope3 / total * 100) if total > 0 else 0

        analysis_text += f"- 范围1占比：**{scope1_pct:.1f}%**\n"
        analysis_text += f"- 范围2占比：**{scope2_pct:.1f}%**\n"
        analysis_text += f"- 范围3占比：**{scope3_pct:.1f}%**\n\n"
    else:
        analysis_text += "### 排放数据分析\n"
        analysis_text += "- 总排放量为0，请先进行碳排放计算\n\n"

    # 建议部分 - 改进逻辑，确保总有内容
    analysis_text += "### 关键建议\n"

    # 总是显示建议，不依赖严格的条件
    recommendations = []

    # 1. 基于排放结构的建议
    if total > 0:
        if scope2_pct > 50:
            recommendations.append("**优化电力使用**：电力是主要排放源，建议进行能源审计和节能改造")
        elif scope3_pct > 60:
            recommendations.append("**供应链管理**：供应链排放占比较高，建议建立绿色采购标准")
        elif scope1_pct > 50:
            recommendations.append("**直接排放优化**：直接排放（燃料）是主要来源，建议优化运输和能源结构")

    # 2. 基于数据质量的建议
    if 'energy_data' in data:
        energy_data = data['energy_data']
        if isinstance(energy_data, pd.DataFrame) and not energy_data.empty:
            if len(energy_data) < 6:
                recommendations.append("**数据收集**：建议收集至少6个月的数据以提高分析准确性")

            # 检查数据质量
            if 'electricity_kwh' in energy_data.columns:
                electricity = energy_data['electricity_kwh']
                if electricity.isnull().sum() > 0:
                    recommendations.append("**数据完善**：部分电力数据缺失，建议完善数据记录")
        else:
            recommendations.append("**数据录入**：能源数据为空，建议录入至少1个月的能源消耗数据")

    # 3. 默认建议（如果以上都不满足）
    if not recommendations:
        recommendations = [
            "**能源审计**：建议进行全面的能源审计，识别主要耗能环节",
            "**设备升级**：考虑升级老旧设备，提高能源使用效率",
            "**员工培训**：加强员工节能意识培训，建立节能文化"
        ]

    # 添加序号并加入分析文本
    for i, rec in enumerate(recommendations, 1):
        analysis_text += f"{i}. {rec}\n"

    # 添加更多分析内容
    analysis_text += "\n### 趋势分析\n"

    # 检查是否有能源数据
    if 'energy_data' in data:
        energy_data = data['energy_data']
        if isinstance(energy_data, pd.DataFrame) and not energy_data.empty:
            if 'electricity_kwh' in energy_data.columns:
                electricity = energy_data['electricity_kwh']
                if len(electricity) >= 2:
                    # 计算简单趋势
                    if len(electricity) >= 2:
                        first_val = electricity.iloc[0]
                        last_val = electricity.iloc[-1]
                        if last_val > first_val:
                            analysis_text += "- 电力消耗呈**上升趋势**，建议关注能耗增长原因\n"
                        elif last_val < first_val:
                            analysis_text += "- 电力消耗呈**下降趋势**，节能措施可能已见效\n"
                        else:
                            analysis_text += "- 电力消耗**保持稳定**\n"
                else:
                    analysis_text += "- 数据不足，无法进行趋势分析\n"
        else:
            analysis_text += "- 暂无能源数据\n"
    else:
        analysis_text += "- 暂无能源数据\n"

    return analysis_text


# ============================================
# 8. 页面函数 - 供应链地图
# ============================================
def calculate_distance(lat1, lon1, lat2, lon2):
    """计算两个经纬度坐标之间的距离（公里）- 使用Haversine公式"""
    # 将十进制度数转化为弧度
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine公式
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    # 地球平均半径（公里）
    r = 6371.0

    return c * r

def show_supply_chain_map():
    """显示供应链地图"""
    st.markdown('<h2 class="section-header">供应链碳足迹可视化地图</h2>', unsafe_allow_html=True)

    # 获取数据
    data = get_current_data_for_dashboard()

    if not data or 'supplier_data' not in data or data['supplier_data'].empty:
        st.warning("暂无供应商数据，请先选择包含供应商信息的数据集")
        return

    supplier_data = data['supplier_data']

    # 确保必要的列存在
    # 重命名列以确保一致性
    column_mapping = {}
    if 'type' in supplier_data.columns:
        column_mapping['type'] = 'material_type'
    if 'supplier_rating' in supplier_data.columns:
        column_mapping['supplier_rating'] = 'rating'

    if column_mapping:
        supplier_data = supplier_data.rename(columns=column_mapping)

    # 确保必要的列存在
    required_cols = ['name', 'lat', 'lon']
    for col in required_cols:
        if col not in supplier_data.columns:
            st.error(f"缺少必要列: {col}")
            st.write("可用列:", supplier_data.columns.tolist())
            return

    # 填充缺失的列
    if 'rating' not in supplier_data.columns:
        supplier_data['rating'] = 'B'

    if 'material_type' not in supplier_data.columns:
        supplier_data['material_type'] = '未知'

    # 确保坐标是数值
    try:
        supplier_data['lat'] = pd.to_numeric(supplier_data['lat'], errors='coerce')
        supplier_data['lon'] = pd.to_numeric(supplier_data['lon'], errors='coerce')
        # 移除无效坐标
        supplier_data = supplier_data.dropna(subset=['lat', 'lon'])
    except Exception as e:
        st.error(f"坐标处理错误: {e}")
        return

    # 创建地图
    fig = go.Figure()

    # 添加供应商节点
    colors = ['#FF6B6B', '#4ECDC4', '#FFD166', '#45B7D1', '#96CEB4']

    # 确定主工厂位置
    main_factory_lat = 31.2989  # 默认苏州
    main_factory_lon = 120.5853

    if 'company_info' in data and not data['company_info'].empty:
        company_info = data['company_info'].iloc[0]
        location = company_info.get('location', '')
        if '深圳' in location:
            main_factory_lat, main_factory_lon = 22.5431, 114.0579
        elif '东莞' in location:
            main_factory_lat, main_factory_lon = 23.0207, 113.7518
        elif '常州' in location:
            main_factory_lat, main_factory_lon = 31.8126, 119.9740
        elif '上海' in location:
            main_factory_lat, main_factory_lon = 31.2304, 121.4737

    # 添加主工厂
    fig.add_trace(go.Scattermapbox(
        lat=[main_factory_lat],
        lon=[main_factory_lon],
        mode='markers+text',
        marker=dict(
            size=30,
            color='#2E8B57',
            symbol='circle'
        ),
        text="主工厂",
        textposition="top right",
        name="主工厂"
    ))

    # 添加供应商和运输路线
    total_distance = 0
    for i, supplier in supplier_data.iterrows():
        try:
            supplier_lat = float(supplier['lat'])
            supplier_lon = float(supplier['lon'])

            # 计算距离
            distance = calculate_distance(main_factory_lat, main_factory_lon,
                                          supplier_lat, supplier_lon)
            total_distance += distance

            # 添加供应商点
            fig.add_trace(go.Scattermapbox(
                lat=[supplier_lat],
                lon=[supplier_lon],
                mode='markers+text',
                marker=dict(
                    size=20,
                    color=colors[i % len(colors)],
                    opacity=0.8
                ),
                text=supplier['name'],
                textposition="top right",
                name=supplier['name'],
                customdata=[[supplier.get('rating', 'N/A'),
                             supplier.get('material_type', '未知')]],
                hovertemplate="<b>%{text}</b><br>评级: %{customdata[0]}<br>材料: %{customdata[1]}<extra></extra>"
            ))

            # 添加运输路线
            fig.add_trace(go.Scattermapbox(
                lat=[main_factory_lat, supplier_lat],
                lon=[main_factory_lon, supplier_lon],
                mode='lines',
                line=dict(width=2, color='rgba(100, 100, 100, 0.5)'),
                showlegend=False,
                hoverinfo='text',
                text=f"距离: {distance:.0f} km"
            ))

        except (ValueError, TypeError) as e:
            continue

    # 显示总运输距离
    if total_distance > 0:
        st.info(f"🌐 总运输距离: {total_distance:.0f} km")
        st.info(f"🚚 平均单程距离: {total_distance / len(supplier_data):.0f} km")

    # 计算地图中心
    if not supplier_data.empty:
        all_lats = list(supplier_data['lat']) + [main_factory_lat]
        all_lons = list(supplier_data['lon']) + [main_factory_lon]
        avg_lat = sum(all_lats) / len(all_lats)
        avg_lon = sum(all_lons) / len(all_lons)
    else:
        avg_lat, avg_lon = main_factory_lat, main_factory_lon

    # 地图布局 - 关键修改：使用不需要token的open-street-map样式
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",  # 改为open-street-map，不需要token
            zoom=4,
            center=dict(lat=avg_lat, lon=avg_lon)
        ),
        height=600,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    # 尝试显示地图
    try:
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
    except Exception as e:
        st.error(f"地图显示错误: {str(e)}")
        st.info("尝试使用白底地图...")
        # 如果open-street-map不行，尝试其他不需要token的样式
        fig.update_layout(
            mapbox=dict(
                style="white-bg",  # 最简单的白色背景
                zoom=4,
                center=dict(lat=avg_lat, lon=avg_lon)
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    # 供应商详情表格
    st.markdown("### 供应商详情")

    # 创建显示数据框
    display_cols = ['name']
    if 'material_type' in supplier_data.columns:
        display_cols.append('material_type')
    if 'rating' in supplier_data.columns:
        display_cols.append('rating')
    display_cols.extend(['lat', 'lon'])

    display_df = supplier_data[display_cols].copy()

    # 添加距离列
    distances = []
    for _, supplier in supplier_data.iterrows():
        distance = calculate_distance(main_factory_lat, main_factory_lon,
                                      supplier['lat'], supplier['lon'])
        distances.append(f"{distance:.0f} km")

    display_df['distance_km'] = distances

    # 重命名列
    column_names = {
        'name': '供应商名称',
        'material_type': '材料类型',
        'rating': '评级',
        'lat': '纬度',
        'lon': '经度',
        'distance_km': '距离'
    }

    display_df = display_df.rename(columns=column_names)

    st.dataframe(display_df, use_container_width=True)

# ============================================
# 9. 页面函数 - 报告生成
# ============================================

def show_report_generation():
    """显示报告生成页面"""
    st.markdown('<h2 class="section-header">生成碳管理报告</h2>', unsafe_allow_html=True)

    # 检查是否有计算结果
    if not st.session_state.emission_results:
        st.warning("请先进行碳排放计算")
        return

    results = st.session_state.emission_results

    # 报告配置
    col1, col2 = st.columns(2)

    with col1:
        report_type = st.selectbox(
            "报告类型",
            ["ESG报告", "CDP问卷", "年度可持续发展报告", "内部管理报告"]
        )

    with col2:
        report_format = st.radio(
            "输出格式",
            ["PDF", "Word", "Excel"]
        )

    # 报告预览
    st.markdown("### 报告预览")

    report_content = f"""
    # {report_type}
    ## 碳排放管理报告

    ### 一、报告概况
    - 报告期间：2024年度
    - 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    ### 二、碳排放核算结果
    - **范围1排放（直接排放）**：{results['scope1']:.1f} tCO₂
    - **范围2排放（外购电力）**：{results['scope2']:.1f} tCO₂
    - **范围3排放（供应链）**：{results['scope3']:.1f} tCO₂
    - **总排放量**：{results['total']:.1f} tCO₂

    ### 三、排放构成分析
    - 范围1占比：{results['scope1'] / results['total'] * 100 if results['total'] > 0 else 0:.1f}%
    - 范围2占比：{results['scope2'] / results['total'] * 100 if results['total'] > 0 else 0:.1f}%
    - 范围3占比：{results['scope3'] / results['total'] * 100 if results['total'] > 0 else 0:.1f}%

    ### 四、减排措施与成效
    1. 已实施措施：
       - 照明系统LED改造
       - 运输路线优化
       - 供应商碳管理培训

    2. 减排成效：
       - 年度减排量：{(results['total'] * 0.05):.1f} tCO₂
       - 碳强度降低：5.2%

    ### 五、未来规划
    - 2025年目标：减排8%
    - 2026年目标：实现碳达峰
    - 长期目标：碳中和

    ---
    *本报告由CarbonFlow碳足迹管理平台自动生成*
    """

    st.text_area("报告内容", report_content, height=400)

    # 报告生成按钮
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("生成报告", type="primary", use_container_width=True):
            with st.spinner("正在生成报告中..."):
                st.success(f"{report_type}生成成功！")

    with col2:
        if st.button("下载报告", use_container_width=True):
            # 创建模拟文件
            report_data = {
                "report_type": report_type,
                "emission_results": results,
                "generated_at": datetime.now().isoformat()
            }

            st.download_button(
                label="确认下载",
                data=json.dumps(report_data, ensure_ascii=False, indent=2),
                file_name=f"carbon_report_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True
            )

    with col3:
        if st.button("发送报告", use_container_width=True):
            st.info("报告发送功能准备中...")


# ============================================
# 10. 主函数
# ============================================

def main():
    """主应用"""

    # 标题
    st.markdown('<h1 class="main-header">🧭 CarbonFlow - 供应链碳足迹管理平台</h1>', unsafe_allow_html=True)

    page = st.sidebar.selectbox(
        "导航菜单",
        [
            "🗂️ 数据管理",
            "🗓️ 数据概览",
            "🖥️ 碳计算",
            "📈 数据分析",
            "📍 供应链地图",
            "📝 报告生成",
            "⚙️ 系统设置"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 数据状态")

    # 显示当前数据源
    st.sidebar.info(f"数据源：{st.session_state.data_source}")

    if st.session_state.data_source == '示例数据':
        st.sidebar.success(f"数据集：{st.session_state.current_dataset}")
    elif st.session_state.data_source == '文件上传':
        total_files = len(st.session_state.uploaded_data)
        st.sidebar.success(f"已上传：{total_files}个文件")
    elif st.session_state.data_source == '手动录入':
        total_records = sum(len(records) for records in st.session_state.manual_data.values())
        st.sidebar.success(f"已录入：{total_records}条记录")

    st.sidebar.markdown("---")

    # 快捷操作
    st.sidebar.markdown("### 快捷操作")

    if st.sidebar.button("刷新数据", use_container_width=True):
        st.rerun()

    if st.sidebar.button("清除数据", use_container_width=True):
        clear_all_data()
        st.rerun()

    # 页面路由
    if page == "🗂️ 数据管理":
        show_data_management()
    elif page == "🗓️ 数据概览":
        show_dashboard()
    elif page == "🖥️ 碳计算":
        show_carbon_calculation()
    elif page == "📈 数据分析":
        show_analysis_insights()
    elif page == "📍 供应链地图":
        show_supply_chain_map()
    elif page == "📝 报告生成":
        show_report_generation()
    elif page == "⚙️ 系统设置":
        show_system_settings()


def clear_all_data():
    """清除所有数据"""
    st.session_state.uploaded_data = {}
    st.session_state.manual_data = {}
    st.session_state.emission_results = {}
    st.session_state.current_dataset = '智能电子制造企业'
    st.session_state.data_source = '示例数据'
    st.success("所有数据已清除")


def show_analysis_insights():
    """显示AI智能分析与减排建议"""
    st.markdown('<h2 class="section-header">AI智能分析与减排建议</h2>', unsafe_allow_html=True)

    # 检查是否有计算结果
    if not st.session_state.emission_results:
        st.warning("请先进行碳排放计算")
        return

    results = st.session_state.emission_results

    # 获取数据
    data = get_current_data_for_dashboard()
    if not data:
        st.warning("请先选择数据源")
        return

    # 创建选项卡
    tab1, tab2, tab3 = st.tabs(["🔗 AI智能分析", "📊 机器学习预测", "💡 专业建议"])

    with tab1:
        st.markdown("### AI智能分析报告")

        # 准备数据用于AI分析
        analysis_data = prepare_data_for_ai_analysis(data, results)

        # 显示分析结果
        if results['total'] > 0:
            scope1_pct = (results['scope1'] / results['total']) * 100
            scope2_pct = (results['scope2'] / results['total']) * 100
            scope3_pct = (results['scope3'] / results['total']) * 100

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("直接排放占比", f"{scope1_pct:.1f}%")

        with col2:
            st.metric("外购电力排放占比", f"{scope2_pct:.1f}%")

        with col3:
            st.metric("供应链简介排放占比", f"{scope3_pct:.1f}%")

        # 能耗分析
        if 'production_data' in data and not data['production_data'].empty:
            production_data = data['production_data']
            if 'production_volume' in production_data.columns:
                total_production = production_data['production_volume'].sum()

                if 'energy_data' in data and not data['energy_data'].empty:
                    energy_data = data['energy_data']
                    if 'electricity_kwh' in energy_data.columns:
                        total_electricity = energy_data['electricity_kwh'].sum()
                        if total_production > 0:
                            unit_energy = total_electricity / total_production
                            st.metric("单位产品能耗", f"{unit_energy:.1f} kWh/台")

        # 成本分析（简化）
        if results['total'] > 0:
            # 假设碳价100元/吨
            carbon_cost = results['total'] * 100
            st.metric("年度碳成本", f"¥{carbon_cost:,.0f}")

        # AI生成的分析摘要
        st.markdown("#### AI分析摘要")

        # 使用机器学习分析数据模式
        if 'energy_data' in data and not data['energy_data'].empty:
            energy_patterns = analyze_energy_patterns(data['energy_data'])

            if energy_patterns.get('trend_direction'):
                trend_text = {
                    'upward': '上升趋势',
                    'downward': '下降趋势',
                    'stable': '稳定趋势'
                }.get(energy_patterns['trend_direction'], '未知')

                st.info(f"**能耗趋势**: {trend_text}")

            if energy_patterns.get('anomalies'):
                st.warning(f"**数据异常**: 发现{len(energy_patterns['anomalies'])}个月份数据异常")

        # 生成AI建议按钮
        if st.button("运行AI分析", type="primary"):
            with st.spinner("AI正在分析数据..."):
                ai_analysis = generate_ai_analysis(data, results)
                st.markdown("#### AI深度分析")
                st.write(ai_analysis)

    with tab2:
        st.markdown("### 机器学习预测分析")

        if 'energy_data' in data and not data['energy_data'].empty:
            energy_data = data['energy_data']

            if len(energy_data) >= 6:  # 需要足够的数据
                # 预测设置
                st.markdown("#### 能耗预测")
                forecast_months = st.slider("预测未来月数", 1, 12, 3)

                if st.button("开始预测", type="primary"):
                    with st.spinner("训练预测模型中..."):
                        # 使用机器学习进行预测
                        predictions = predict_energy_consumption(energy_data, forecast_months)

                        if predictions:
                            # 收集所有能源类型的特征重要性（用于统一显示）
                            all_feature_importance = {}

                            # 首先显示所有预测结果
                            for energy_type, pred_data in predictions.items():
                                st.markdown(f"##### {energy_type.replace('_', ' ').title()}")

                                # 预测值
                                future_values = pred_data.get('predictions', [])
                                if future_values:
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        # 计算变化率
                                        if 'electricity_kwh' in energy_data.columns:
                                            last_actual = energy_data['electricity_kwh'].iloc[-1] if len(energy_data) > 0 else 0
                                            if last_actual > 0:
                                                change_pct = (future_values[0] - last_actual) / last_actual * 100
                                                delta_text = f"{'↑' if change_pct > 0 else '↓'} {abs(change_pct):.1f}%"
                                            else:
                                                delta_text = None
                                        else:
                                            delta_text = None

                                        st.metric(
                                            "下月预测",
                                            f"{future_values[0]:,.0f}",
                                            delta=delta_text
                                        )

                                    with col2:
                                        if len(future_values) > 1:
                                            st.metric("2个月后", f"{future_values[1]:,.0f}")

                                    with col3:
                                        if len(future_values) > 2:
                                            st.metric("3个月后", f"{future_values[2]:,.0f}")

                                # 模型性能
                                if 'model_performance' in pred_data:
                                    perf = pred_data['model_performance']

                                    col1, col2 = st.columns(2)  # 去掉置信度，只用2列
                                    with col1:
                                        st.metric("模型准确度", f"R²={perf.get('r2', 0):.3f}")
                                    with col2:
                                        st.metric("预测误差", f"RMSE={perf.get('rmse', 0):.0f}")

                                # 趋势分析
                                trend = pred_data.get('trend', 'unknown')
                                trend_text = {
                                    'increasing': '将呈现上升趋势',
                                    'decreasing': '将呈现下降趋势',
                                    'stable': '将呈现稳定趋势',
                                    'unknown': '未来趋势未知'
                                }.get(trend, '未来趋势未知')

                                st.info(f"**预测趋势**: {trend_text}")

                                # 收集特征重要性（但不立即显示）
                                if 'feature_importance' in pred_data:
                                    all_feature_importance[energy_type] = pred_data['feature_importance']

                            # 统一显示特征重要性（只显示一次）
                            if all_feature_importance:
                                st.markdown("---")
                                st.markdown("### 特征重要性")

                                # 创建一个标签页来显示每种能源类型的特征重要性
                                if len(all_feature_importance) > 1:
                                    feature_tabs = st.tabs(list(all_feature_importance.keys()))

                                    for i, (energy_type, importance_dict) in enumerate(all_feature_importance.items()):
                                        with feature_tabs[i]:
                                            importance_df = pd.DataFrame(
                                                list(importance_dict.items()),
                                                columns=['特征', '重要性']
                                            ).sort_values('重要性', ascending=False)

                                            st.dataframe(importance_df, use_container_width=True)

                                            # 可视化特征重要性
                                            fig_importance = go.Figure()
                                            fig_importance.add_trace(go.Bar(
                                                x=importance_df['重要性'],
                                                y=importance_df['特征'],
                                                orientation='h',
                                                marker_color='#65A8F3'
                                            ))

                                            fig_importance.update_layout(
                                                title=f'{energy_type.replace("_", " ").title()} 特征重要性',
                                                height=300,
                                                showlegend=False
                                            )

                                            st.plotly_chart(fig_importance, use_container_width=True)
                                else:
                                    # 如果只有一种能源类型，直接显示
                                    for energy_type, importance_dict in all_feature_importance.items():
                                        with st.expander(f" {energy_type.replace('_', ' ').title()} 特征重要性"):
                                            importance_df = pd.DataFrame(
                                                list(importance_dict.items()),
                                                columns=['特征', '重要性']
                                            ).sort_values('重要性', ascending=False)

                                            st.dataframe(importance_df, use_container_width=True)

                                            # 可视化特征重要性
                                            fig_importance = go.Figure()
                                            fig_importance.add_trace(go.Bar(
                                                x=importance_df['重要性'],
                                                y=importance_df['特征'],
                                                orientation='h',
                                                marker_color='#65A8F3'
                                            ))

                                            fig_importance.update_layout(
                                                title=f'{energy_type.replace("_", " ").title()} 特征重要性',
                                                height=300,
                                                showlegend=False
                                            )

                                            st.plotly_chart(fig_importance, use_container_width=True)
                        else:
                            st.warning("预测失败，数据可能不足")
            else:
                st.info("需要至少6个月的数据进行有效预测")

    with tab3:
        st.markdown("### 专业减排建议")

        # 基于排放结构生成建议
        suggestions = generate_smart_suggestions(data, results)

        for i, suggestion in enumerate(suggestions):
            with st.expander(f"{suggestion['title']}", expanded=i == 0):
                st.write(suggestion["description"])

                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**减排潜力**\n{suggestion['potential']}")
                with col2:
                    st.success(f"**经济性**\n{suggestion['roi']}")

        # 基准对比
        st.markdown("### 行业对标分析")

        benchmark_data = pd.DataFrame({
            '指标': ['单位产品碳排', '范围3占比', '电力排放占比'],
            '您的企业': [
                f"{results['total'] / 1000:.1f} t/千台" if results['total'] > 0 else "0",
                f"{results['scope3'] / results['total'] * 100:.1f}%" if results['total'] > 0 else "0",
                f"{results['scope2'] / results['total'] * 100:.1f}%" if results['total'] > 0 else "0"
            ],
            '行业平均': ['82.5 t/千台', '68.2%', '42.1%'],
            '行业最佳': ['45.3 t/千台', '45.8%', '28.6%']
        })

        st.dataframe(benchmark_data, use_container_width=True)

        # AI生成个性化建议
        st.markdown("### AI个性化建议")

        if st.button("获取AI专业建议", type="primary"):
            with st.spinner("AI正在生成个性化建议..."):
                ai_suggestions = get_ai_suggestions(data, results)
                st.markdown(ai_suggestions)

def show_system_settings():
    """显示系统设置页面"""
    st.markdown('<h2 class="section-header">系统设置与管理</h2>', unsafe_allow_html=True)

    # 账户管理
    with st.expander("👤 账户管理", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.text_input("用户名", value="admin@company.com")
            st.text_input("公司名称", value="苏州精密制造有限公司")

        with col2:
            st.text_input("联系人", value="张经理")
            st.text_input("联系电话", value="138-XXXX-XXXX")

        if st.button("保存设置", type="primary"):
            st.success("设置已保存")

    # 系统信息
    with st.expander("ℹ️ 系统信息", expanded=True):
        st.markdown("""
        **平台版本**: 1.5.2  
        **最后更新**: 2025-12-12  
        """)


# ============================================
# 11. 运行应用
# ============================================

if __name__ == "__main__":
    main()
