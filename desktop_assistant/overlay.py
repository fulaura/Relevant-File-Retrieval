from __future__ import annotations

import os
import webbrowser
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

DocRecord = Dict[str, object]


class ResultsOverlay(QtWidgets.QDialog):
	chat_requested = QtCore.Signal()

	def __init__(self) -> None:
		super().__init__()
		self.setWindowTitle("AITU Retrieval Overlay")
		self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.WindowCloseButtonHint)
		self.resize(720, 520)
		self._last_docs: List[DocRecord] = []
		self._last_ocr: str = ""
		self._temp_top_hint = False
		self._focus_timer = QtCore.QTimer(self)
		self._focus_timer.setSingleShot(True)
		self._focus_timer.timeout.connect(self._remove_temp_top_hint)
		self._build_ui()

	def _build_ui(self) -> None:
		main_layout = QtWidgets.QVBoxLayout(self)

		self.status_label = QtWidgets.QLabel("Ready")
		self.status_label.setObjectName("status-label")
		main_layout.addWidget(self.status_label)

		self.ocr_preview = QtWidgets.QPlainTextEdit()
		self.ocr_preview.setReadOnly(True)
		self.ocr_preview.setPlaceholderText("OCR snippet will appear here.")
		main_layout.addWidget(self.ocr_preview)

		self.table = QtWidgets.QTableWidget(0, 3)
		self.table.setHorizontalHeaderLabels(["Similarity", "Document", "Path"])
		header = self.table.horizontalHeader()
		header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
		header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
		header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
		self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
		self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
		self.table.doubleClicked.connect(self._open_selected_path)
		main_layout.addWidget(self.table)

		btn_row = QtWidgets.QHBoxLayout()
		btn_row.addStretch()

		self.chat_btn = QtWidgets.QPushButton("Open Chat")
		self.chat_btn.clicked.connect(self.chat_requested.emit)
		btn_row.addWidget(self.chat_btn)

		close_btn = QtWidgets.QPushButton("Close")
		close_btn.clicked.connect(self.hide)
		btn_row.addWidget(close_btn)

		main_layout.addLayout(btn_row)

		shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Q"), self)
		shortcut.activated.connect(self.hide)
		self._close_shortcut = shortcut

	def display_results(self, ocr_text: str, docs: List[DocRecord]) -> None:
		self._last_docs = docs
		self._last_ocr = ocr_text

		self.ocr_preview.setPlainText(ocr_text.strip())
		self.table.setRowCount(len(docs))

		for row, doc in enumerate(docs):
			sim = doc.get("chunks", [{}])[0].get("similarity")
			similarity_val = f"{sim:.4f}" if isinstance(sim, float) else "-"
			sim_item = QtWidgets.QTableWidgetItem(similarity_val)
			file_item = QtWidgets.QTableWidgetItem(str(doc.get("file_name", "Unknown")))
			path_item = QtWidgets.QTableWidgetItem(str(doc.get("path", "")))

			for item in (sim_item, file_item, path_item):
				item.setData(QtCore.Qt.UserRole, doc)

			self.table.setItem(row, 0, sim_item)
			self.table.setItem(row, 1, file_item)
			self.table.setItem(row, 2, path_item)

		summary_text = f"Found {len(docs)} document(s)." if docs else "No documents matched this capture."
		self.status_label.setText(summary_text)
		self._present()

	def show_processing(self, message: str) -> None:
		self.status_label.setText(message)
		self.ocr_preview.clear()
		self.table.setRowCount(0)
		self._present()

	def context_summary(self) -> str:
		lines: List[str] = []
		if self._last_ocr:
			snippet = self._last_ocr.strip().replace("\n", " ")
			lines.append(f"OCR snippet: {snippet[:400]}")
		else:
			lines.append("OCR snippet: <empty>")

		if not self._last_docs:
			lines.append("No matching documents were found.")
		else:
			lines.append("Matching documents:")
			for doc in self._last_docs[:5]:
				path = doc.get("path", "")
				name = doc.get("file_name", "Unknown")
				sim = doc.get("chunks", [{}])[0].get("similarity")
				score = f" score={sim:.4f}" if isinstance(sim, float) else ""
				lines.append(f"- {name} ({path}){score}")
		return "\n".join(lines)

	def _present(self) -> None:
		if not self._temp_top_hint:
			self._apply_temp_top_hint()
		if not self.isVisible():
			self.show()
		self.raise_()
		self.activateWindow()
		if self._temp_top_hint:
			self._focus_timer.start(600)

	def _apply_temp_top_hint(self) -> None:
		self._temp_top_hint = True
		window = self.windowHandle()
		if window is not None:
			window.setFlag(QtCore.Qt.WindowStaysOnTopHint, True)
		else:
			self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)

	def _open_selected_path(self) -> None:
		items = self.table.selectedItems()
		if not items:
			return
		doc = items[0].data(QtCore.Qt.UserRole)
		path = doc.get("path") if isinstance(doc, dict) else None
		if not path:
			return
		path_obj = Path(path)
		if path_obj.exists():
			os.startfile(path_obj)
		else:
			webbrowser.open(str(path_obj))

	def _remove_temp_top_hint(self) -> None:
		if not self._temp_top_hint:
			return
		window = self.windowHandle()
		if window is not None:
			window.setFlag(QtCore.Qt.WindowStaysOnTopHint, False)
		else:
			self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, False)
		self._temp_top_hint = False


class _ChatWorker(QtCore.QObject):
	finished = QtCore.Signal(str)
	failed = QtCore.Signal(str)

	def __init__(self, send_fn: Callable[[str], str], message: str) -> None:
		super().__init__()
		self._send_fn = send_fn
		self._message = message

	@QtCore.Slot()
	def run(self) -> None:
		try:
			reply = self._send_fn(self._message)
		except Exception as exc:  # noqa: BLE001
			self.failed.emit(str(exc))
		else:
			self.finished.emit(reply)


class ChatDialog(QtWidgets.QDialog):
	def __init__(self, send_fn: Callable[[str], str], parent: Optional[QtWidgets.QWidget] = None) -> None:
		super().__init__(parent)
		self.setWindowTitle("AITU Document Chat")
		self.resize(600, 420)
		self.send_fn = send_fn
		self._context_info: dict | None = None
		self._active_threads: List[QtCore.QThread] = []
		self._thread_workers: Dict[QtCore.QThread, _ChatWorker] = {}
		self._is_busy = False
		self._build_ui()

	def _build_ui(self) -> None:
		layout = QtWidgets.QVBoxLayout(self)

		self.context_button = QtWidgets.QPushButton("Context")
		self.context_button.setCheckable(True)
		self.context_button.hide()
		self.context_button.toggled.connect(self._toggle_context_details)
		layout.addWidget(self.context_button)

		self.context_tree = QtWidgets.QTreeWidget()
		self.context_tree.setHeaderLabels(["Document", "Similarity", "Path"])
		header = self.context_tree.header()
		header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
		header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
		header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
		self.context_tree.hide()
		self.context_tree.itemDoubleClicked.connect(self._open_context_doc)
		layout.addWidget(self.context_tree)

		self.history = QtWidgets.QTextEdit()
		self.history.setReadOnly(True)
		layout.addWidget(self.history)

		self.input = QtWidgets.QPlainTextEdit()
		self.input.setPlaceholderText("Ask follow-up questions...")
		self.input.setFixedHeight(100)
		layout.addWidget(self.input)

		self.send_btn = QtWidgets.QPushButton("Send")
		self.send_btn.clicked.connect(self._on_send)
		layout.addWidget(self.send_btn)

		shortcut_return = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Return"), self)
		shortcut_return.activated.connect(self._on_send)
		shortcut_enter = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Enter"), self)
		shortcut_enter.activated.connect(self._on_send)
		self._send_shortcuts = [shortcut_return, shortcut_enter]

		close_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Q"), self)
		close_shortcut.activated.connect(self.close)
		self._close_shortcut = close_shortcut

	def _on_send(self) -> None:
		if self._is_busy:
			return
		message = self.input.toPlainText().strip()
		if not message:
			return
		self.history.append(f"<b>You:</b> {message}")
		self.input.clear()
		self.history.ensureCursorVisible()
		self._set_busy(True)
		self._start_worker(message)

	def _start_worker(self, message: str) -> None:
		thread = QtCore.QThread(self)
		worker = _ChatWorker(self.send_fn, message)
		worker.moveToThread(thread)
		thread.started.connect(worker.run)
		worker.finished.connect(self._handle_reply)
		worker.failed.connect(self._handle_error)
		worker.finished.connect(thread.quit)
		worker.failed.connect(thread.quit)
		worker.finished.connect(worker.deleteLater)
		worker.failed.connect(worker.deleteLater)
		thread.finished.connect(thread.deleteLater)
		thread.finished.connect(lambda: self._cleanup_thread(thread))
		self._active_threads.append(thread)
		self._thread_workers[thread] = worker
		thread.start()

	def _handle_reply(self, reply: str) -> None:
		self.history.append(f"<b>Assistant:</b> {reply}")
		self.history.ensureCursorVisible()
		self._set_busy(False)

	def _handle_error(self, error: str) -> None:
		self.history.append(f"<b>Assistant:</b> Error: {error}")
		self.history.ensureCursorVisible()
		self._set_busy(False)

	def _cleanup_thread(self, thread: QtCore.QThread) -> None:
		if thread in self._active_threads:
			self._active_threads.remove(thread)
		self._thread_workers.pop(thread, None)

	def _set_busy(self, busy: bool) -> None:
		self._is_busy = busy
		self.send_btn.setEnabled(not busy)
		self.send_btn.setText("Thinking..." if busy else "Send")

	def update_context(self, context: dict | None) -> None:
		self._context_info = context
		docs = (context or {}).get("docs", [])
		if not docs:
			self.context_button.hide()
			self.context_tree.hide()
			self.context_button.setChecked(False)
			self.context_tree.clear()
			return

		count = context.get("count", len(docs))
		label = f"{count} file{'s' if count != 1 else ''}"
		self.context_button.setText(label)
		self.context_button.show()

		self.context_tree.clear()
		for entry in docs:
			name = entry.get("name", "Unknown")
			sim = entry.get("similarity")
			sim_text = f"{sim:.4f}" if isinstance(sim, float) else "-"
			path = entry.get("path", "")
			item = QtWidgets.QTreeWidgetItem([name, sim_text, path])
			item.setData(0, QtCore.Qt.UserRole, path)
			self.context_tree.addTopLevelItem(item)

		if self.context_button.isChecked():
			self.context_tree.show()

	def _toggle_context_details(self, checked: bool) -> None:
		if checked and self.context_tree.topLevelItemCount():
			self.context_tree.show()
		else:
			self.context_tree.hide()

	def _open_context_doc(self, item: QtWidgets.QTreeWidgetItem, _column: int) -> None:
		path = item.data(0, QtCore.Qt.UserRole)
		if not isinstance(path, str) or not path:
			return
		path_obj = Path(path)
		if path_obj.exists():
			os.startfile(path_obj)
		else:
			webbrowser.open(str(path_obj))
