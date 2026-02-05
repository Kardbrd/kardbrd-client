"""Unit tests for directory-based state management."""

import threading

from kardbrd_client import BoardSubscription, DirectoryStateManager


class TestDirectoryStateManager:
    """Unit tests for DirectoryStateManager."""

    def test_create_directory_if_not_exists(self, tmp_path):
        """Directory is created on first write."""
        state_dir = tmp_path / "state"
        manager = DirectoryStateManager(str(state_dir))
        sub = BoardSubscription(
            board_id="test-board",
            api_url="http://test",
            bot_token="tok",
            agent_name="Test",
        )
        manager.add_subscription(sub)
        assert state_dir.exists()
        assert (state_dir / "test-board.json").exists()

    def test_empty_directory_returns_no_subscriptions(self, tmp_path):
        """Empty directory returns empty dict, not error."""
        manager = DirectoryStateManager(str(tmp_path))
        assert manager.get_all_subscriptions() == {}
        assert manager.has_subscriptions() is False

    def test_add_and_get_subscription(self, tmp_path):
        """Add subscription and retrieve it."""
        manager = DirectoryStateManager(str(tmp_path))
        sub = BoardSubscription(
            board_id="board-1",
            api_url="http://test",
            bot_token="tok",
            agent_name="Test",
        )
        manager.add_subscription(sub)
        retrieved = manager.get_subscription("board-1")
        assert retrieved is not None
        assert retrieved.board_id == "board-1"
        assert retrieved.api_url == "http://test"

    def test_get_nonexistent_subscription_returns_none(self, tmp_path):
        """Get nonexistent subscription returns None."""
        manager = DirectoryStateManager(str(tmp_path))
        assert manager.get_subscription("nonexistent") is None

    def test_remove_subscription_deletes_file(self, tmp_path):
        """Remove subscription deletes the board file."""
        manager = DirectoryStateManager(str(tmp_path))
        sub = BoardSubscription(
            board_id="board-1",
            api_url="http://test",
            bot_token="tok",
            agent_name="Test",
        )
        manager.add_subscription(sub)
        assert (tmp_path / "board-1.json").exists()
        result = manager.remove_subscription("board-1")
        assert result is True
        assert not (tmp_path / "board-1.json").exists()

    def test_remove_nonexistent_subscription_returns_false(self, tmp_path):
        """Remove nonexistent subscription returns False."""
        manager = DirectoryStateManager(str(tmp_path))
        result = manager.remove_subscription("nonexistent")
        assert result is False

    def test_get_all_subscriptions(self, tmp_path):
        """Get all subscriptions returns all boards."""
        manager = DirectoryStateManager(str(tmp_path))
        for i in range(3):
            sub = BoardSubscription(
                board_id=f"board-{i}",
                api_url="http://test",
                bot_token="tok",
                agent_name=f"Test{i}",
            )
            manager.add_subscription(sub)

        subscriptions = manager.get_all_subscriptions()
        assert len(subscriptions) == 3
        assert "board-0" in subscriptions
        assert "board-1" in subscriptions
        assert "board-2" in subscriptions

    def test_has_subscriptions(self, tmp_path):
        """has_subscriptions returns correct state."""
        manager = DirectoryStateManager(str(tmp_path))
        assert manager.has_subscriptions() is False

        sub = BoardSubscription(
            board_id="board-1",
            api_url="http://test",
            bot_token="tok",
            agent_name="Test",
        )
        manager.add_subscription(sub)
        assert manager.has_subscriptions() is True

    def test_concurrent_updates_different_boards(self, tmp_path):
        """Concurrent updates to different boards don't block each other."""
        manager = DirectoryStateManager(str(tmp_path))
        results = []

        def update_board(board_id):
            sub = BoardSubscription(
                board_id=board_id,
                api_url="http://test",
                bot_token="tok",
                agent_name="Test",
            )
            manager.add_subscription(sub)
            manager.update_board_activity(board_id, f"activity-{board_id}", "2025-01-01T00:00:00")
            results.append(board_id)

        threads = [threading.Thread(target=update_board, args=(f"board-{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert len(list(tmp_path.glob("board-*.json"))) == 10

    def test_active_tasks_per_board(self, tmp_path):
        """Active tasks are stored per-board."""
        manager = DirectoryStateManager(str(tmp_path))
        sub1 = BoardSubscription(
            board_id="board-1",
            api_url="http://test",
            bot_token="tok",
            agent_name="Test",
        )
        sub2 = BoardSubscription(
            board_id="board-2",
            api_url="http://test",
            bot_token="tok",
            agent_name="Test",
        )
        manager.add_subscription(sub1)
        manager.add_subscription(sub2)

        manager.update_active_task("board-1", "task-1", {"status": "running"})
        manager.update_active_task("board-2", "task-2", {"status": "pending"})

        assert manager.get_active_tasks("board-1") == {"task-1": {"status": "running"}}
        assert manager.get_active_tasks("board-2") == {"task-2": {"status": "pending"}}

    def test_remove_active_task(self, tmp_path):
        """Remove active task from board."""
        manager = DirectoryStateManager(str(tmp_path))
        sub = BoardSubscription(
            board_id="board-1",
            api_url="http://test",
            bot_token="tok",
            agent_name="Test",
        )
        manager.add_subscription(sub)
        manager.update_active_task("board-1", "task-1", {"status": "running"})
        manager.update_active_task("board-1", "task-2", {"status": "pending"})

        manager.remove_active_task("board-1", "task-1")
        tasks = manager.get_active_tasks("board-1")
        assert "task-1" not in tasks
        assert "task-2" in tasks

    def test_update_board_activity(self, tmp_path):
        """Update activity tracking for a board."""
        manager = DirectoryStateManager(str(tmp_path))
        sub = BoardSubscription(
            board_id="board-1",
            api_url="http://test",
            bot_token="tok",
            agent_name="Test",
        )
        manager.add_subscription(sub)
        manager.update_board_activity("board-1", "act-123", "2025-01-15T10:00:00")

        retrieved = manager.get_subscription("board-1")
        assert retrieved is not None
        assert retrieved.last_activity_id == "act-123"
        assert retrieved.last_activity_time == "2025-01-15T10:00:00"

    def test_get_since_timestamp(self, tmp_path):
        """Get since timestamp returns last activity time."""
        manager = DirectoryStateManager(str(tmp_path))
        sub = BoardSubscription(
            board_id="board-1",
            api_url="http://test",
            bot_token="tok",
            agent_name="Test",
        )
        manager.add_subscription(sub)

        # Initially None
        assert manager.get_since_timestamp("board-1") is None

        # After update
        manager.update_board_activity("board-1", "act-123", "2025-01-15T10:00:00")
        assert manager.get_since_timestamp("board-1") == "2025-01-15T10:00:00"

    def test_reload_returns_subscriptions(self, tmp_path):
        """Reload returns all subscriptions from disk."""
        manager = DirectoryStateManager(str(tmp_path))
        sub = BoardSubscription(
            board_id="board-1",
            api_url="http://test",
            bot_token="tok",
            agent_name="Test",
        )
        manager.add_subscription(sub)

        # Reload returns subscriptions
        subscriptions = manager.reload()
        assert "board-1" in subscriptions
        assert subscriptions["board-1"].agent_name == "Test"
