from collections.abc import Sequence
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel
from sqlalchemy import (
    delete as sqlalchemy_delete,
    insert,
    select,
    update as sqlalchemy_update,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement
from loguru import logger

from .base_model import Base


T = TypeVar("T", bound=Base)


class BaseDAO(Generic[T]):
    model: type[T] = None

    def __init__(self, session: AsyncSession):
        self._session = session
        if self.model is None:
            raise ValueError(
                "The target SQLAlchemy model must be explicitly defined in the child DAO class."
            )

    async def find_one_or_none_by_id(self, data_id: Any) -> T | None:
        try:
            record = await self._session.get(self.model, data_id)
            return record
        except SQLAlchemyError:
            logger.exception(
                f"Database error during primary key lookup for {self.model.__name__} (ID: {data_id})"
            )
            raise

    async def find_all(
        self,
        filters: BaseModel | dict[str, Any] | None = None,
        expressions: list[ColumnElement[bool]] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        order_by: Any | None = None,
    ) -> Sequence[T]:
        """Don't use it with relationships. Be careful with big result sets."""
        try:
            query = select(self.model)

            if filters:
                filter_dict = (
                    filters.model_dump(exclude_unset=True)
                    if isinstance(filters, BaseModel)
                    else filters
                )
                query = query.filter_by(**filter_dict)

            if expressions:
                query = query.where(*expressions)

            if order_by is not None:
                query = query.order_by(order_by)
            if limit is not None:
                query = query.limit(limit)
            if offset is not None:
                query = query.offset(offset)

            result = await self._session.execute(query)
            return result.scalars().all()
        except SQLAlchemyError:
            logger.exception(
                f"Database error during dynamic sequence retrieval for {self.model.__name__}"
            )
            raise

    async def add(self, values: BaseModel | dict[str, Any], flush: bool = False) -> T:
        values_dict = (
            values.model_dump(exclude_unset=True)
            if isinstance(values, BaseModel)
            else values
        )
        try:
            new_instance = self.model(**values_dict)
            self._session.add(new_instance)
            if flush:
                await self._session.flush()
                await self._session.refresh(new_instance)
            return new_instance
        except SQLAlchemyError:
            logger.exception(
                f"Database error during singular insertion for {self.model.__name__}"
            )
            raise

    async def add_many(
        self,
        instances: list[BaseModel | dict[str, Any]],
        return_objects: bool = False,
    ) -> list[T] | None:
        # Be careful with default fields
        if not instances:
            return None

        values_list = [
            v.model_dump() if isinstance(v, BaseModel) else v for v in instances
        ]

        try:
            if return_objects:
                stmt = insert(self.model).returning(self.model)
                result = await self._session.execute(stmt, values_list)
                return result.scalars().all()
            else:
                stmt = insert(self.model)
                await self._session.execute(stmt, values_list)
                return None
        except SQLAlchemyError:
            logger.exception(
                f"Database error during core bulk insertion for {self.model.__name__}"
            )
            raise

    async def update(
        self,
        filters: BaseModel | dict[str, Any],
        values: BaseModel | dict[str, Any],
        synchronize_session: Literal["fetch", "evaluate", False] = False,
    ) -> int:
        filter_dict = (
            filters.model_dump(exclude_unset=True)
            if isinstance(filters, BaseModel)
            else filters
        )
        values_dict = (
            values.model_dump(exclude_unset=True)
            if isinstance(values, BaseModel)
            else values
        )

        if not filter_dict:
            raise ValueError(
                "Execution halted: At least one filter criteria is required to prevent accidental full-table mutations."
            )

        try:
            stmt = (
                sqlalchemy_update(self.model)
                .filter_by(**filter_dict)
                .values(**values_dict)
                .execution_options(synchronize_session=synchronize_session)
            )
            result = await self._session.execute(stmt)
            return result.rowcount
        except SQLAlchemyError:
            logger.exception(
                f"Database error during mutation for {self.model.__name__}"
            )
            raise

    async def delete(
        self,
        filters: BaseModel | dict[str, Any],
        synchronize_session: Literal["fetch", "evaluate", False] = False,
    ) -> int:
        filter_dict = (
            filters.model_dump(exclude_unset=True)
            if isinstance(filters, BaseModel)
            else filters
        )

        if not filter_dict:
            raise ValueError(
                "Execution halted: At least one filter criteria is required to prevent accidental full-table truncations."
            )

        try:
            stmt = (
                sqlalchemy_delete(self.model)
                .filter_by(**filter_dict)
                .execution_options(synchronize_session=synchronize_session)
            )
            result = await self._session.execute(stmt)
            return result.rowcount
        except SQLAlchemyError:
            logger.exception(
                f"Database error during deletion for {self.model.__name__}"
            )
            raise
