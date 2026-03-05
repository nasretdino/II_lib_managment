from collections.abc import Sequence

from sqlalchemy import func

from src.db import BaseDAO
from .models import User
from .schemas import UserFilter


class UserDAO(BaseDAO[User]):
    model = User

    async def find_filtered(
        self,
        filters: UserFilter,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[User]:
        """Case-insensitive partial-match filtering for name."""
        expressions = []
        if filters.name:
            expressions.append(func.lower(User.name).contains(filters.name.lower()))
        return await self.find_all(
            expressions=expressions if expressions else None,
            limit=limit,
            offset=offset,
        )
