from decimal import Decimal
from typing import Any, Dict

from pydantic import ConfigDict, Field, SecretStr

from hummingbot.client.config.config_data_types import BaseConnectorConfigMap
from hummingbot.connector.utils import split_hb_trading_pair
from hummingbot.core.data_type.trade_fee import TradeFeeSchema


DEFAULT_FEES = TradeFeeSchema(
    maker_percent_fee_decimal=Decimal("0.0002"),
    taker_percent_fee_decimal=Decimal("0.0005"),
)
CENTRALIZED = True
EXAMPLE_PAIR = "BTC-USDT"


def is_exchange_information_valid(exchange_info: Dict[str, Any]) -> bool:
    return (
        exchange_info.get("instType") == "SWAP"
        and exchange_info.get("ctType") == "linear"
        and exchange_info.get("state") == "live"
    )


def is_linear_perpetual(trading_pair: str) -> bool:
    _, quote_asset = split_hb_trading_pair(trading_pair)
    return quote_asset in ["USDT", "USDC"]


def get_next_funding_timestamp(current_timestamp: float) -> float:
    current = int(current_timestamp)
    eight_hours = 8 * 60 * 60
    return float(current - current % eight_hours + eight_hours)


class OkxPerpetualConfigMap(BaseConnectorConfigMap):
    connector: str = "okx_perpetual"
    okx_perpetual_api_key: SecretStr = Field(
        default=...,
        json_schema_extra={"prompt": "Enter your Okx Perpetual API key", "is_secure": True, "is_connect_key": True, "prompt_on_new": True},
    )
    okx_perpetual_secret_key: SecretStr = Field(
        default=...,
        json_schema_extra={"prompt": "Enter your Okx Perpetual secret key", "is_secure": True, "is_connect_key": True, "prompt_on_new": True},
    )
    okx_perpetual_passphrase: SecretStr = Field(
        default=...,
        json_schema_extra={"prompt": "Enter your Okx Perpetual passphrase", "is_secure": True, "is_connect_key": True, "prompt_on_new": True},
    )
    model_config = ConfigDict(title="okx_perpetual")


KEYS = OkxPerpetualConfigMap.model_construct()

OTHER_DOMAINS = ["okx_perpetual_demo"]
OTHER_DOMAINS_PARAMETER = {"okx_perpetual_demo": "okx_perpetual_demo"}
OTHER_DOMAINS_EXAMPLE_PAIR = {"okx_perpetual_demo": "BTC-USDT"}
OTHER_DOMAINS_DEFAULT_FEES = {"okx_perpetual_demo": DEFAULT_FEES}


class OkxPerpetualDemoConfigMap(BaseConnectorConfigMap):
    connector: str = "okx_perpetual_demo"
    okx_perpetual_demo_api_key: SecretStr = Field(
        default=...,
        json_schema_extra={"prompt": "Enter your OKX Demo API key", "is_secure": True, "is_connect_key": True, "prompt_on_new": True},
    )
    okx_perpetual_demo_secret_key: SecretStr = Field(
        default=...,
        json_schema_extra={"prompt": "Enter your OKX Demo secret key", "is_secure": True, "is_connect_key": True, "prompt_on_new": True},
    )
    okx_perpetual_demo_passphrase: SecretStr = Field(
        default=...,
        json_schema_extra={"prompt": "Enter your OKX Demo passphrase", "is_secure": True, "is_connect_key": True, "prompt_on_new": True},
    )
    model_config = ConfigDict(title="okx_perpetual_demo")


OTHER_DOMAINS_KEYS = {
    "okx_perpetual_demo": OkxPerpetualDemoConfigMap.model_construct()
}
