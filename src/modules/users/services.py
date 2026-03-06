from datetime import datetime, timezone

from loguru import logger

from src.core.exceptions import NotFoundError

from .dao import UserDAO
from .models import User
from .schemas import UserCreate, UserFilter, UserUpdate


class UserService:
    def __init__(self, dao: UserDAO):
        self.dao = dao

    async def get_all(
        self,
        filters: UserFilter,
        limit: int = 100,
        offset: int = 0,
    ) -> list[User]:
        logger.debug("Fetching users with filters={}", filters)
        return list(
            await self.dao.find_filtered(filters=filters, limit=limit, offset=offset)
        )

    async def get_by_id(self, user_id: int) -> User | None:
        return await self.dao.find_one_or_none_by_id(user_id)

    async def create(self, data: UserCreate) -> User:
        logger.info("Creating user name={}", data.name)
        return await self.dao.add(data, flush=True)

    async def update(self, user_id: int, data: UserUpdate) -> User | None:
        values = data.model_dump(exclude_unset=True)
        if not values:
            # No fields to update — return existing record or 404
            return await self.dao.find_one_or_none_by_id(user_id)

        values["updated_at"] = datetime.now(timezone.utc)
        updated_count = await self.dao.update(
            filters={"id": user_id},
            values=values,
        )
        if updated_count == 0:
            return None
        logger.info("Updated user id={}", user_id)
        return await self.dao.find_one_or_none_by_id(user_id)

    async def delete(self, user_id: int) -> bool:
        deleted_count = await self.dao.delete(filters={"id": user_id})
        if deleted_count > 0:
            logger.info("Deleted user id={}", user_id)
        return deleted_count > 0
