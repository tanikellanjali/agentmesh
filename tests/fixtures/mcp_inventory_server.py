"""A real MCP server standing in for a supplier/inventory system."""
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("inventory")

STOCK = {"apples": 120, "milk": 40, "bread": 8, "lettuce": 0}
SUPPLIERS = {"apples": "FreshFarms", "milk": "DairyCo", "bread": "BakeryPlus", "lettuce": "FreshFarms"}


@mcp.tool()
def stock_level(item: str) -> dict:
    """Return the current stock level for an item."""
    return {"item": item, "units": STOCK.get(item), "supplier": SUPPLIERS.get(item)}


@mcp.tool()
def reorder_list(threshold: int = 10) -> dict:
    """List items at or below a reorder threshold, with their supplier."""
    low = {k: v for k, v in STOCK.items() if v <= threshold}
    return {
        "threshold": threshold,
        "items": [{"item": k, "units": v, "supplier": SUPPLIERS[k]} for k, v in low.items()],
    }


if __name__ == "__main__":
    mcp.run()
