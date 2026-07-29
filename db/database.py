import aiosqlite
from contextlib import asynccontextmanager
from config import settings


@asynccontextmanager
async def get_conn():
    async with aiosqlite.connect(settings.sqlite_path) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode=WAL")
        yield db


async def init_tables() -> None:
    async with aiosqlite.connect(settings.sqlite_path) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS confidence (
                user_id    TEXT NOT NULL,
                topic      TEXT NOT NULL,
                score      REAL DEFAULT 0.0,
                updated_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (user_id, topic)
            );

            CREATE TABLE IF NOT EXISTS training_pairs (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id           TEXT NOT NULL,
                topic             TEXT NOT NULL DEFAULT 'general',
                user_msg          TEXT NOT NULL,
                assistant_msg     TEXT NOT NULL,
                model_used        TEXT NOT NULL,
                engagement_signal TEXT DEFAULT 'continue',
                perplexity        REAL DEFAULT 0.0,
                used_in_training  INTEGER DEFAULT 0,
                created_at        TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS topic_history (
                user_id     TEXT NOT NULL,
                topic       TEXT NOT NULL,
                week_number INTEGER NOT NULL,
                count       INTEGER DEFAULT 0,
                PRIMARY KEY (user_id, topic, week_number)
            );

            CREATE TABLE IF NOT EXISTS checkpoints (
                user_id    TEXT NOT NULL,
                lane       TEXT NOT NULL DEFAULT 'qwen',
                path       TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (user_id, created_at)
            );

            CREATE TABLE IF NOT EXISTS user_skills (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     TEXT NOT NULL,
                skill_name  TEXT NOT NULL,
                trigger     TEXT,
                procedure   TEXT NOT NULL,
                user_prefs  TEXT,
                source_week TEXT NOT NULL,
                active      INTEGER DEFAULT 1,
                created_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS user_rules (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      TEXT NOT NULL,
                rule_text    TEXT NOT NULL,
                applies_to   TEXT DEFAULT 'all',
                confidence   TEXT DEFAULT 'medium',
                source_month TEXT NOT NULL,
                active       INTEGER DEFAULT 1,
                created_at   TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_tp_user
                ON training_pairs(user_id, used_in_training);
            CREATE INDEX IF NOT EXISTS idx_skills_user
                ON user_skills(user_id, active);
            CREATE INDEX IF NOT EXISTS idx_rules_user
                ON user_rules(user_id, active);

            CREATE TABLE IF NOT EXISTS daily_checkins (
                user_id    TEXT NOT NULL,
                date       TEXT NOT NULL,
                questions  TEXT NOT NULL DEFAULT '[]',
                answers    TEXT NOT NULL DEFAULT '[]',
                created_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (user_id, date)
            );

            CREATE TABLE IF NOT EXISTS users (
                id         TEXT PRIMARY KEY,
                email      TEXT UNIQUE NOT NULL,
                username   TEXT UNIQUE NOT NULL,
                password   TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS practice_reps (
                id              TEXT PRIMARY KEY,
                user_id         TEXT NOT NULL,
                date            TEXT NOT NULL,
                observation     TEXT NOT NULL,
                rep_title       TEXT NOT NULL,
                rep_instruction TEXT NOT NULL,
                arc_label       TEXT,
                created_at      TEXT DEFAULT (datetime('now')),
                UNIQUE (user_id, date)
            );

            CREATE TABLE IF NOT EXISTS practice_log (
                user_id   TEXT NOT NULL,
                rep_id    TEXT NOT NULL,
                date      TEXT NOT NULL,
                done      INTEGER DEFAULT 1,
                note      TEXT,
                logged_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (user_id, rep_id)
            );

            CREATE TABLE IF NOT EXISTS twin_sessions (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                question    TEXT NOT NULL,
                response_a  TEXT NOT NULL,
                response_b  TEXT NOT NULL,
                a_is_clone  INTEGER NOT NULL,
                chosen      TEXT,
                created_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS fcm_tokens (
                user_id    TEXT NOT NULL,
                token      TEXT NOT NULL,
                platform   TEXT DEFAULT 'android',
                updated_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (user_id)
            );

            CREATE INDEX IF NOT EXISTS idx_practice_reps_user
                ON practice_reps(user_id, date);
            CREATE INDEX IF NOT EXISTS idx_practice_log_user
                ON practice_log(user_id, date);

            CREATE TABLE IF NOT EXISTS echo_interruptions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    TEXT NOT NULL,
                statement  TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_interruptions_user
                ON echo_interruptions(user_id, created_at);

            CREATE TABLE IF NOT EXISTS echo_revelations (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    TEXT NOT NULL,
                letter     TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_revelations_user
                ON echo_revelations(user_id, created_at);

            CREATE TABLE IF NOT EXISTS echo_threads (
                id               TEXT PRIMARY KEY,
                user_id          TEXT NOT NULL,
                name             TEXT NOT NULL,
                topic            TEXT,
                first_seen       TEXT DEFAULT (datetime('now')),
                last_seen        TEXT DEFAULT (datetime('now')),
                evidence_count   INTEGER DEFAULT 0,
                escalation_level INTEGER DEFAULT 1,
                status           TEXT DEFAULT 'active',
                resolution_note  TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_threads_user_status
                ON echo_threads(user_id, status);

            CREATE TABLE IF NOT EXISTS thread_evidence (
                id              TEXT PRIMARY KEY,
                thread_id       TEXT NOT NULL,
                message_snippet TEXT,
                added_at        TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_evidence_thread
                ON thread_evidence(thread_id, added_at);

            CREATE TABLE IF NOT EXISTS thread_escalations (
                id            TEXT PRIMARY KEY,
                thread_id     TEXT NOT NULL,
                level         INTEGER,
                content       TEXT,
                user_response TEXT,
                shown_at      TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_escalations_thread
                ON thread_escalations(thread_id, shown_at);

            CREATE TABLE IF NOT EXISTS echo_events (
                id           TEXT PRIMARY KEY,
                user_id      TEXT NOT NULL,
                event_type   TEXT NOT NULL,
                source       TEXT NOT NULL DEFAULT 'system',
                summary      TEXT,
                payload_json TEXT NOT NULL DEFAULT '{}',
                weight       REAL DEFAULT 1.0,
                created_at   TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_echo_events_user_time
                ON echo_events(user_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_echo_events_user_type
                ON echo_events(user_id, event_type, created_at);

            CREATE TABLE IF NOT EXISTS life_events (
                id             TEXT PRIMARY KEY,
                user_id        TEXT NOT NULL,
                event_domain   TEXT NOT NULL DEFAULT 'echo',
                event_type     TEXT NOT NULL,
                source         TEXT NOT NULL DEFAULT 'system',
                subject_type   TEXT,
                subject_id     TEXT,
                title          TEXT,
                summary        TEXT,
                payload_json   TEXT NOT NULL DEFAULT '{}',
                confidence     REAL DEFAULT 0.5,
                privacy_level  TEXT DEFAULT 'local',
                created_at     TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_life_events_user_time
                ON life_events(user_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_life_events_user_domain
                ON life_events(user_id, event_domain, event_type, created_at);

            CREATE TABLE IF NOT EXISTS echo_interventions (
                id                  TEXT PRIMARY KEY,
                user_id             TEXT NOT NULL,
                kind                TEXT NOT NULL,
                title               TEXT NOT NULL,
                body                TEXT,
                reason              TEXT NOT NULL,
                source_event_id     TEXT,
                action_type         TEXT NOT NULL DEFAULT 'open_today',
                action_payload_json TEXT NOT NULL DEFAULT '{}',
                priority            INTEGER DEFAULT 1,
                status              TEXT NOT NULL DEFAULT 'pending',
                scheduled_for       TEXT DEFAULT (datetime('now')),
                expires_at          TEXT,
                created_at          TEXT DEFAULT (datetime('now')),
                delivered_at        TEXT,
                acknowledged_at     TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_echo_interventions_user_status
                ON echo_interventions(user_id, status, scheduled_for);
            CREATE INDEX IF NOT EXISTS idx_echo_interventions_user_time
                ON echo_interventions(user_id, created_at);

            CREATE TABLE IF NOT EXISTS intervention_settings (
                user_id         TEXT PRIMARY KEY,
                enabled         INTEGER DEFAULT 1,
                morning_enabled INTEGER DEFAULT 1,
                evening_enabled INTEGER DEFAULT 1,
                clone_enabled   INTEGER DEFAULT 1,
                training_enabled INTEGER DEFAULT 1,
                proof_enabled   INTEGER DEFAULT 1,
                quiet_start     INTEGER DEFAULT 22,
                quiet_end       INTEGER DEFAULT 8,
                max_per_day     INTEGER DEFAULT 3,
                updated_at      TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS clone_missions (
                id               TEXT PRIMARY KEY,
                user_id          TEXT NOT NULL,
                run_id           TEXT NOT NULL,
                candidate_id     TEXT NOT NULL,
                winning_style    TEXT NOT NULL,
                strategy         TEXT NOT NULL,
                suggested_action TEXT NOT NULL,
                risk             TEXT,
                outcome_question TEXT,
                practice_rep_id  TEXT,
                status           TEXT DEFAULT 'active',
                created_at       TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_clone_missions_user_time
                ON clone_missions(user_id, created_at);

            CREATE TABLE IF NOT EXISTS shadow_outcomes (
                id           TEXT PRIMARY KEY,
                user_id      TEXT NOT NULL,
                event_id     TEXT,
                subject_type TEXT NOT NULL,
                subject_id   TEXT,
                outcome      TEXT NOT NULL,
                score        REAL DEFAULT 0.0,
                note         TEXT,
                created_at   TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_shadow_outcomes_user_time
                ON shadow_outcomes(user_id, created_at);

            CREATE TABLE IF NOT EXISTS proof_items (
                id               TEXT PRIMARY KEY,
                user_id          TEXT NOT NULL,
                title            TEXT NOT NULL,
                description      TEXT,
                category         TEXT DEFAULT 'practice',
                source_type      TEXT,
                source_id        TEXT,
                evidence         TEXT,
                skill_tags_json  TEXT NOT NULL DEFAULT '[]',
                opportunity_type TEXT DEFAULT 'personal_goal',
                status           TEXT DEFAULT 'active',
                created_at       TEXT DEFAULT (datetime('now')),
                updated_at       TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_proof_items_user_time
                ON proof_items(user_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_proof_items_user_category
                ON proof_items(user_id, category, created_at);

            CREATE TABLE IF NOT EXISTS opportunity_goals (
                id                  TEXT PRIMARY KEY,
                user_id             TEXT NOT NULL,
                title               TEXT NOT NULL,
                type                TEXT DEFAULT 'personal_goal',
                description         TEXT,
                required_proof_json TEXT NOT NULL DEFAULT '[]',
                missing_proof_json  TEXT NOT NULL DEFAULT '[]',
                next_step           TEXT,
                status              TEXT DEFAULT 'suggested',
                created_at          TEXT DEFAULT (datetime('now')),
                updated_at          TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_opportunity_goals_user_status
                ON opportunity_goals(user_id, status, created_at);

            CREATE TABLE IF NOT EXISTS tournament_runs (
                id            TEXT PRIMARY KEY,
                user_id       TEXT NOT NULL,
                prompt        TEXT NOT NULL,
                topic         TEXT DEFAULT 'general',
                status        TEXT DEFAULT 'candidate',
                winning_style TEXT,
                created_at    TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_tournament_runs_user_time
                ON tournament_runs(user_id, created_at);

            CREATE TABLE IF NOT EXISTS tournament_candidates (
                id           TEXT PRIMARY KEY,
                run_id       TEXT NOT NULL,
                style        TEXT NOT NULL,
                response     TEXT NOT NULL,
                score        REAL DEFAULT 0.0,
                signals_json TEXT NOT NULL DEFAULT '{}',
                created_at   TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_tournament_candidates_run
                ON tournament_candidates(run_id);

            CREATE TABLE IF NOT EXISTS user_theses (
                id                       TEXT PRIMARY KEY,
                user_id                  TEXT NOT NULL,
                title                    TEXT NOT NULL,
                statement                TEXT NOT NULL,
                stage                    TEXT DEFAULT 'forming',
                status                   TEXT DEFAULT 'active',
                confidence               REAL DEFAULT 0.0,
                evidence_count           INTEGER DEFAULT 0,
                next_action_type         TEXT DEFAULT 'none',
                next_action_label        TEXT DEFAULT 'Keep talking',
                next_action_payload_json TEXT NOT NULL DEFAULT '{}',
                created_at               TEXT DEFAULT (datetime('now')),
                updated_at               TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_user_theses_user_status
                ON user_theses(user_id, status, updated_at);

            CREATE TABLE IF NOT EXISTS thesis_evidence (
                id           TEXT PRIMARY KEY,
                thesis_id    TEXT NOT NULL,
                user_id      TEXT NOT NULL,
                source       TEXT NOT NULL,
                subject_type TEXT NOT NULL,
                subject_id   TEXT,
                summary      TEXT NOT NULL,
                weight       REAL DEFAULT 1.0,
                created_at   TEXT DEFAULT (datetime('now')),
                UNIQUE(thesis_id, source, subject_type, subject_id, summary)
            );
            CREATE INDEX IF NOT EXISTS idx_thesis_evidence_thesis
                ON thesis_evidence(thesis_id, created_at);

            CREATE TABLE IF NOT EXISTS teacher_usage (
                id            TEXT PRIMARY KEY,
                user_id       TEXT NOT NULL,
                purpose       TEXT NOT NULL,
                reason        TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at    TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_teacher_usage_user_time
                ON teacher_usage(user_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_teacher_usage_user_purpose
                ON teacher_usage(user_id, purpose, created_at);

            CREATE TABLE IF NOT EXISTS training_runs (
                id              TEXT PRIMARY KEY,
                user_id         TEXT NOT NULL,
                lane            TEXT NOT NULL DEFAULT 'gemma4_e2b',
                status          TEXT NOT NULL DEFAULT 'running',
                untrained_pairs INTEGER DEFAULT 0,
                required_pairs  INTEGER DEFAULT 0,
                adapter_path    TEXT,
                error           TEXT,
                summary_json    TEXT NOT NULL DEFAULT '{}',
                started_at      TEXT DEFAULT (datetime('now')),
                finished_at     TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_training_runs_user_lane_time
                ON training_runs(user_id, lane, started_at);

            CREATE TABLE IF NOT EXISTS training_lock (
                resource     TEXT PRIMARY KEY,
                run_id       TEXT NOT NULL,
                user_id      TEXT NOT NULL,
                lane         TEXT NOT NULL,
                acquired_at  TEXT DEFAULT (datetime('now'))
            );
        """)
        async with db.execute("PRAGMA table_info(checkpoints)") as cur:
            checkpoint_cols = {row[1] for row in await cur.fetchall()}
        if "lane" not in checkpoint_cols:
            await db.execute("ALTER TABLE checkpoints ADD COLUMN lane TEXT NOT NULL DEFAULT 'qwen'")
        await db.commit()
