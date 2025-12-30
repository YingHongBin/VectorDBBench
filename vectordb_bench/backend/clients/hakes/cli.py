from typing import Annotated, TypedDict, Unpack

import click

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    get_custom_case_config,
    run,
)
from .. import DB
from .config import HAKESConfig, HAKESCaseConfig


class HAKESTypedDict(TypedDict):
    host: Annotated[
        str,
        click.option("--host", type=str, help="HAKES host address", required=True),
    ]
    port: Annotated[
        int,
        click.option("--port", type=int, default=8080, help="HAKES port", show_default=True),
    ]


class HAKESCaseTypedDict(CommonTypedDict, HAKESTypedDict):
    nprobe: Annotated[
        int,
        click.option("--nprobe", type=int, help="Number of clusters to probe during search", required=True),
    ]
    k_factor: Annotated[
        int,
        click.option("--k-factor", type=int, help="K factor for search", required=True),
    ]
    metric_type: Annotated[
        str,
        click.option(
            "--metric-type",
            type=click.Choice(["L2", "IP"], case_sensitive=False),
            help="Distance metric type",
            default="IP",
            show_default=True,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(HAKESCaseTypedDict)
def HAKES(**parameters: Unpack[HAKESCaseTypedDict]):
    """Run benchmark tests for HAKES vector database"""
    
    # Add custom case configuration
    parameters["custom_case"] = get_custom_case_config(parameters)
    
    run(
        db=DB.HAKES,
        db_config=HAKESConfig(
            db_label=parameters["db_label"],
            host=parameters["host"],
            port=parameters["port"],
        ),
        db_case_config=HAKESCaseConfig(
            k=parameters["k"],
            nprobe=parameters["nprobe"],
            k_factor=parameters["k_factor"],
            metric_type=parameters["metric_type"],
        ),
        **parameters,
    )
