"""Initial schema with users, datasets, jobs, and results tables

Revision ID: b421c157c655
Revises:
Create Date: 2025-11-13 14:32:12.909508

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'b421c157c655'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create ENUM types
    user_role_enum = postgresql.ENUM('user', 'researcher', 'admin', name='user_role', create_type=True)
    user_role_enum.create(op.get_bind())

    data_type_enum = postgresql.ENUM(
        'bulk_rna_seq', 'single_cell_rna_seq', 'methylation', 'copy_number',
        'wes', 'wgs', 'proteomics', 'metabolomics', 'radiomics', 'clinical',
        name='data_type', create_type=True
    )
    data_type_enum.create(op.get_bind())

    job_type_enum = postgresql.ENUM(
        'feature_selection', 'model_training', 'benchmarking',
        'preprocessing', 'visualization',
        name='job_type', create_type=True
    )
    job_type_enum.create(op.get_bind())

    job_status_enum = postgresql.ENUM(
        'pending', 'running', 'completed', 'failed', 'cancelled',
        name='job_status', create_type=True
    )
    job_status_enum.create(op.get_bind())

    # Create users table
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('email', sa.String(255), unique=True, nullable=False, index=True),
        sa.Column('username', sa.String(100), unique=True, nullable=False, index=True),
        sa.Column('hashed_password', sa.String(255), nullable=False),
        sa.Column('full_name', sa.String(255), nullable=True),
        sa.Column('role', user_role_enum, nullable=False, server_default='user'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True, onupdate=sa.func.now()),
    )

    # Create datasets table
    op.create_table(
        'datasets',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('data_type', data_type_enum, nullable=False),
        sa.Column('file_path', sa.String(1024), nullable=True),
        sa.Column('n_samples', sa.Integer(), nullable=True),
        sa.Column('n_features', sa.Integer(), nullable=True),
        sa.Column('metadata_json', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('owner_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True, onupdate=sa.func.now()),
    )
    op.create_index('ix_datasets_owner_id', 'datasets', ['owner_id'])

    # Create jobs table
    op.create_table(
        'jobs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('job_type', job_type_enum, nullable=False),
        sa.Column('status', job_status_enum, nullable=False, server_default='pending'),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('dataset_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('datasets.id', ondelete='CASCADE'), nullable=False),
        sa.Column('celery_task_id', sa.String(255), nullable=True),
        sa.Column('config', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('result_id', postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_index('ix_jobs_user_id', 'jobs', ['user_id'])
    op.create_index('ix_jobs_dataset_id', 'jobs', ['dataset_id'])
    op.create_index('ix_jobs_status', 'jobs', ['status'])
    op.create_index('ix_jobs_celery_task_id', 'jobs', ['celery_task_id'])

    # Create results table
    op.create_table(
        'results',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('job_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('jobs.id', ondelete='CASCADE'), nullable=False, unique=True),
        sa.Column('selected_features', postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column('metrics', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('artifacts_path', sa.String(1024), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_results_job_id', 'results', ['job_id'])

    # Add foreign key for result_id in jobs table (circular reference)
    op.create_foreign_key(
        'fk_jobs_result_id',
        'jobs', 'results',
        ['result_id'], ['id'],
        ondelete='SET NULL'
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Drop foreign key constraint
    op.drop_constraint('fk_jobs_result_id', 'jobs', type_='foreignkey')

    # Drop tables in reverse order
    op.drop_index('ix_results_job_id', 'results')
    op.drop_table('results')

    op.drop_index('ix_jobs_celery_task_id', 'jobs')
    op.drop_index('ix_jobs_status', 'jobs')
    op.drop_index('ix_jobs_dataset_id', 'jobs')
    op.drop_index('ix_jobs_user_id', 'jobs')
    op.drop_table('jobs')

    op.drop_index('ix_datasets_owner_id', 'datasets')
    op.drop_table('datasets')

    op.drop_table('users')

    # Drop ENUM types
    sa.Enum(name='job_status').drop(op.get_bind(), checkfirst=False)
    sa.Enum(name='job_type').drop(op.get_bind(), checkfirst=False)
    sa.Enum(name='data_type').drop(op.get_bind(), checkfirst=False)
    sa.Enum(name='user_role').drop(op.get_bind(), checkfirst=False)
