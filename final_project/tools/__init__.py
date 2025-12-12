# tools/__init__.py
from .tool_registry import ToolRegistry
from .search_tool import get_search_tool_spec
from .calc_tool import get_calc_tool_spec
from .time_tool import get_time_tool_spec
from .rag_tool import get_rag_tool_spec
from .memory_tools import get_read_memory_tool_spec, get_write_memory_tool_spec


def register_default_tools() -> ToolRegistry:
    """
    프로젝트에서 기본으로 사용할 Tool들을 한 번에 등록.
    - web_search
    - calculator
    - get_time
    - rag_search
    - read_memory
    - write_memory
    """
    reg = ToolRegistry()
    reg.register_tool(get_search_tool_spec())
    reg.register_tool(get_calc_tool_spec())
    reg.register_tool(get_time_tool_spec())
    reg.register_tool(get_rag_tool_spec())
    reg.register_tool(get_read_memory_tool_spec())
    reg.register_tool(get_write_memory_tool_spec())
    return reg
